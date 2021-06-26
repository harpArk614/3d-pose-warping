import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi
import cv2

def warpping_module(volume, masks, transform):
    n, h, w, d, c = volume.get_shape().as_list()
    arr = {}
    arr['masks'] = masks

    count = transform.shape[1]

    vol_inv = tf.linalg.LinearOperatorInversion( volume, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None,is_square=None, name=None)
    affine_mul = tf.matmul(transform, vol_inv)

    """affine_mul = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]"""

    affine_mul = np.array(affine_mul).reshape((1, 1, 4, 4))
    affine_transforms = transform * affine_mul
    affine_transforms = tf.reshape(affine_transforms, (-1, 4, 4))

    expanded_tensor = np.expand_dims(volume, -1)
    tiled_tensor = tf.tile(expanded_tensor, multiples=[1, count, 1, 1, 1, 1])
    repeated_tensor = tf.reshape(tiled_tensor, (n * count, h, w, d, c))

    transposed_masks = tf.transpose(masks, perm=[0, 4, 1, 2, 3])
    reshaped_masks = tf.reshape(transposed_masks, [n * count, h, w, d])
    repeated_tensor = repeated_tensor * np.expand_dims(reshaped_masks, axis=-1)

    arr['masked_bodyparts'] = repeated_tensor

    warped = tfg.math.interpolation.trilinear.interpolate( repeated_tensor, affine_transforms, name='trilinear_interpolate')
    arr['masks_warped'] = warped

    res = tf.reshape(warped, [-1, count, h, w, d, c])
    res = tf.transpose(res, [0, 2, 3, 4, 1, 5])
    
    res = tf.reduce_max(res, reduction_indices=[-2])

    return res, arr
  
