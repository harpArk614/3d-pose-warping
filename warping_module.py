import tensorflow as tf
from parameters import params
import numpy as np
from utils import extend_spatial_sizes, reduce_spatial_sizes

def build_coords(shape):
    xx, yy, zz = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]), tf.range(shape[2]))  # in image notation
    ww = tf.ones(xx.shape)
    coords = tf.concat([tf.expand_dims(tf.cast(a, tf.float32), -1) for a in [xx, yy, zz, ww]], axis=-1)
    return coords


# input in matrix notation
def transform_single(volume, transform, interpolation):
    volume = tf.transpose(volume, [1, 0, 2, 3])  # switch to image notation
    coords = build_coords(volume.shape[:3])
    coords_shape = coords.shape
    coords_reshaped = tf.reshape(coords, [-1, 4])
    pointers_reshaped = tf.linalg.matmul(transform, coords_reshaped, transpose_b=True)
    pointers = tf.reshape(tf.transpose(pointers_reshaped, [1, 0]), coords_shape)  # undo transpose_b
    pointers = pointers[:, :, :, :3]
    if interpolation == 'NEAREST':
        pointers = tf.cast(tf.math.round(pointers), dtype=tf.int32)
        with tf.device('/gpu:0'):
            res = tf.gather_nd(volume, pointers)
    elif interpolation == 'TRILINEAR':
        c3s = {}
        for c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            c3s[c] = tf.gather_nd(volume, tf.cast(tf.floor(pointers), dtype=tf.int32) + c)
        d = pointers - tf.floor(pointers)
        c2s = {}
        for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            c2s[c] = c3s[(0,) + c] * (1 - d[:, :, :, 0:1]) + c3s[(1,) + c] * (d[:, :, :, 0:1])
        c1s = {}
        for c in [(0,), (1,)]:
            c1s[c] = c2s[(0,) + c] * (1 - d[:, :, :, 1:2]) + c2s[(1,) + c] * (d[:, :, :, 1:2])
        res = c1s[(0,)] * (1 - d[:, :, :, 2:3]) + c1s[(1,)] * (d[:, :, :, 2:3])
    else:
        raise ValueError
    return res


def volumetric_transform(volumes, transforms, interpolation='NEAREST'):
    return tf.map_fn(lambda x: transform_single(x[0], x[1], interpolation), (volumes, transforms), dtype=tf.float32,
                     parallel_iterations=128)


def warp_3d(vol_batch, masks_batch, transform_batch, reduce=True):
    n, h, w, d, c = vol_batch.get_shape().as_list()
    with tf.name_scope('warp_3d'):
        net = {}

        part_count = transform_batch.shape[1]

        net['bodypart_masks'] = masks_batch

        init_volume_size = (params['image_size'], params['image_size'], params['image_size'])
        z_scale = (d - 1) / (h - 1)
        v_scale = (h - 1) / init_volume_size[0]
        affine_mul = [[1, 1, 1 / z_scale, v_scale],
                      [1, 1, 1 / z_scale, v_scale],
                      [z_scale, z_scale, 1, v_scale * z_scale],
                      [1, 1, 1 / z_scale, 1]]
        affine_mul = np.array(affine_mul).reshape((1, 1, 4, 4))
        affine_transforms = transform_batch * affine_mul
        affine_transforms = tf.reshape(affine_transforms, (-1, 4, 4))

        expanded_tensor = tf.expand_dims(vol_batch, -1)
        multiples = [1, part_count, 1, 1, 1, 1]
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, (
            n * part_count, h, w, d, c))

        transposed_masks = tf.transpose(masks_batch, [0, 4, 1, 2, 3])
        reshaped_masks = tf.reshape(transposed_masks, [n * part_count, h, w, d])
        repeated_tensor = repeated_tensor * tf.expand_dims(reshaped_masks, axis=-1)

        net['masked_bodyparts'] = repeated_tensor
        warped = volumetric_transform(repeated_tensor, affine_transforms, interpolation='TRILINEAR')
        net['masked_bodyparts_warped'] = warped

        res = tf.reshape(warped, [-1, part_count, h, w, d, c])
        res = tf.transpose(res, [0, 2, 3, 4, 1, 5])
        if reduce:
            res = tf.reduce_max(res, reduction_indices=[-2])
        return res, net


def tf_pose_map_3d(poses, shape):
    y = tf.unstack(poses, axis=1)
    y[0], y[1] = y[1], y[0]
    poses = tf.stack(y, axis=1)
    image_size = tf.constant(params['image_size'], tf.float32)
    shape = tf.constant(shape, tf.float32)
    sigma = tf.constant(6, tf.float32)
    poses = tf.unstack(poses, axis=0)
    pose_mapss = []
    for pose in poses:
        pose = pose / image_size * shape[:, tf.newaxis]
        joints = tf.unstack(pose, axis=-1)
        pose_maps = []
        for joint in joints:
            xx, yy, zz = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
            mesh = tf.cast(tf.stack([xx, yy, zz]), dtype=tf.float32)
            pose_map = mesh - joint[:, tf.newaxis, tf.newaxis, tf.newaxis]
            pose_map = pose_map / shape[:, tf.newaxis, tf.newaxis, tf.newaxis] * image_size
            pose_map = tf.exp(-tf.reduce_sum(pose_map ** 2, axis=0) / (2 * sigma ** 2))
            pose_maps.append(pose_map)
        pose_map = tf.stack(pose_maps, axis=-1)
        if params['2d_3d_pose']:
            pose_map = tf.reduce_max(pose_map, axis=2, keepdims=True)
            pose_map = tf.tile(pose_map, [1, 1, params['depth'], 1])
        pose_mapss.append(pose_map)
    return tf.stack(pose_mapss, axis=0)
