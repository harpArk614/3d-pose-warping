from dataset_definitions import get_dataset
from model import generator, discriminator
from parallel_map import parallel_map_as_tf_dataset
from losses import init_perception_model, get_pose_loss, init_pose_model
import tensorflow as tf
from utils import initialize_uninitialized, make_pretrained_weight_loader, ssim
from io import BytesIO
import matplotlib.pyplot as plt
import time
from parameters import params
import numpy as np
import tensorflow_gan as tfgan

backend = tf.keras.backend

if __name__ == '__main__':
    print('Hyperparams:')
    for name, val in params.items():
        print('{}:\t{}'.format(name, val))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    backend.set_session(sess)
    init_perception_model()

    # VALIDATION GRAPH
    print('build validation graph')

    # the validation dataset consists of the same samples every time, so results are comparable
    valid_count = params['valid_count']

    valid_dataset = get_dataset(params['dataset'], deterministic=True, with_to_masks=True)
    valid_data = []
    if params['with_valid']:  # if we train with valid, we use the test set instead of the valid set for validation
        for valid_sample in valid_dataset.next_test_sample():
            valid_data.append(valid_sample)
            if len(valid_data) == valid_count:
                break
    else:
        for valid_sample in valid_dataset.next_valid_sample():
            valid_data.append(valid_sample)
            if len(valid_data) == valid_count:
                break

    def valid_gen():
        while True:
            for sample in valid_data:
                yield sample


    valid_dataset = parallel_map_as_tf_dataset(None, valid_gen(), n_workers=1, deterministic=True)
    valid_dataset = valid_dataset.batch(1, drop_remainder=True)
    valid_iterator = valid_dataset.make_one_shot_iterator()
    (valid_img_from, valid_img_to, valid_mask_from, valid_mask_to, valid_transform_params, valid_3d_input_pose,
      valid_3d_target_pose) = valid_iterator.get_next()

    print('- GAN')
    with tf.variable_scope('GAN', reuse=False):
        pose_gan = tfgan.gan_model(
            generator,
            discriminator,
            real_data= valid_img_to,
            generator_inputs=[valid_img_from, valid_mask_from, valid_transform_params, valid_3d_input_pose,
                              valid_3d_target_pose],
            check_shapes=False
        )

        # 2D mask for target pose to compute foreground SSIM
    if params['2d_3d_warp']:
        valid_fg_mask = valid_mask_to
    else:
        valid_fg_mask = tf.reduce_max(valid_mask_to, axis=3)
    valid_fg_mask = valid_fg_mask[:, :-1, :-1]
    valid_fg_mask = tf.image.resize_images(valid_fg_mask, (params['image_size'], params['image_size']),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    valid_fg_mask = tf.reduce_max(valid_fg_mask, axis=3)

    with tf.variable_scope('GAN/Generator', reuse=True):
        valid_model = pose_gan.generator_fn([valid_img_from, valid_mask_from, valid_transform_params, valid_3d_input_pose, valid_3d_target_pose])

    valid_pose_loss = get_pose_loss(valid_img_to, valid_model[0])

    init_pose_model(sess, 'pose3d_minimal/checkpoint/model.ckpt-160684')

    start = time.time()
    checkpoint = tf.train.latest_checkpoint(params['check_dir'])
    summary_writer = tf.summary.FileWriter(params['tb_dir'])
    if checkpoint is not None:
        start_step = int(checkpoint.split('-')[-1])
        init_fn = make_pretrained_weight_loader(checkpoint, 'GAN', 'GAN', ['Adam', 'Momentum'], [])
        init_fn(sess)
        global_step = tf.Variable(start_step, trainable=False, name='global_step')
        initialize_uninitialized(sess)
        print(f'Loaded checkpoint from step {start_step}:', time.time() - start)

        print('Performing validation')
        val_start = time.time()
        v_inp = []
        v_tar = []
        v_gen = []
        v_pl = []
        v_bg = []
        v_bg_mask = []
        v_fg = []
        v_fg_m = []
        valid_generated = tf.clip_by_value(valid_model[0], -1, 1)
        print('- generating images')
        for _ in range(valid_count):
            inp, tar, gen, pl, bg, bg_mask, fg, fg_m = sess.run(
                [valid_img_from, valid_img_to, valid_generated, valid_pose_loss, valid_model[1]['background'],
                  valid_model[1]['foreground_mask'], valid_model[1]['foreground'], valid_fg_mask])
            v_inp.append(inp[0, :256, :256] / 2 + .5)
            v_tar.append(tar[0, :256, :256] / 2 + .5)
            v_gen.append(gen[0, :256, :256] / 2 + .5)
            v_pl += [pl]
            v_bg.append(bg[0, :256, :256] / 2 + .5)
            v_bg_mask.append(np.tile(bg_mask[0, :256, :256], [1, 1, 3]))
            v_fg.append(fg[0, :256, :256] / 2 + .5)
            v_fg_m.append(fg_m[0, ..., np.newaxis])

        prefix = 'test' if params['with_valid'] else 'val'
        print('- computing SSIM scores')
        ssim_score, ssim_fg, ssim_bg = ssim(v_tar, v_gen, masks=v_fg_m)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim', simple_value=ssim_score)])
        summary_writer.add_summary(summary, 0)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim_fg', simple_value=ssim_fg)])
        summary_writer.add_summary(summary, 0)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim_bg', simple_value=ssim_bg)])
        summary_writer.add_summary(summary, 0)

        print('- computing pose score')
        pl = np.mean(v_pl)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/pose_loss', simple_value=pl)])
        summary_writer.add_summary(summary, 0)

        print('- creating images for tensorboard')
        v_inp = np.concatenate(v_inp[:16], axis=0)
        v_tar = np.concatenate(v_tar[:16], axis=0)
        v_gen = np.concatenate(v_gen[:16], axis=0)
        v_bg = np.concatenate(v_bg[:16], axis=0)
        v_bg_mask = np.concatenate(v_bg_mask[:16], axis=0)
        v_fg = np.concatenate(v_fg[:16], axis=0)
        res = np.concatenate([v_inp, v_tar, v_gen, v_bg, v_bg_mask, v_fg], axis=1)
        plt.imsave('output/res_with_mask.png', res, format='png')
        s = BytesIO()
        plt.imsave(s, res, format='png')
        res = tf.Summary.Image(encoded_image_string=s.getvalue(), height=res.shape[0], width=res.shape[1])
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_img', image=res)])
        summary_writer.add_summary(summary, 0)
        summary_writer.flush()
        print('Performed validation:', time.time() - val_start)

        res2 = np.concatenate([v_inp, v_tar, v_gen], axis=1)
        plt.imsave('output/res1.png', res2, format='png')

    else:
        print("No Model Found!!")

