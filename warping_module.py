import tensorflow as tf
import numpy as np
from scipy import ndimage as ndi
import cv2

def warping_module(volume, masks, transform):
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
"""
def rotation_estimation(joint1i,joint2i,joint1f,joint2f,mask):
  
  a=joint1i-joint2i
  b=joint1f-joint2f
  
  
  scle=np.linalg.norm([b[0],b[1]])/np.linalg.norm([a[0],a[1]])
  
  angle=tf.math.atan(b[1]/b[0])-tf.math.atan(a[1]/a[0])
  si=tf.math.sin(angle)
  co=tf.math.cos(angle)
  
  rotation_mat=np.array([[co,-si],[si,co]])
  
  mxy=np.array([mask[0,:],mask[1,:]])
  j2i=np.array([[joint2i[0]],[joint2i[1]]])
  j2f=np.array([[joint2f[0]],[joint2f[1]]])
  
  xymask=np.matmul(rotation_mat,mxy-j2i)*scle+j2f
  
  
  zmask=((xymask[0,:]-joint2f[0])*b[2]/b[0])+joint2f[2]
  
  zmask=np.reshape(zmask,(1,-1))
  maskf=np.concatenate((xymask,zmask),axis=0)
  
  return maskf

def warpingModule(mask,transform,joint):
    warped_mask=[]
    lefthand_upper  =   rotation_estimation(joint['lsho'],joint['lelb'],transform['lsho'],transform['lelb'],mask[' '])
    righthand_upper =   rotation_estimation(joint['rsho'],joint['relb'],transform['rsho'],transform['relb'],mask[' '])
    lefthand_lower  =   rotation_estimation(joint['lelb'],joint['lwri'],transform['lelb'],transform['lwri'],mask[' '])
    righthand_lower =   rotation_estimation(joint['relb'],joint['rwri'],transform['relb'],transform['rwri'],mask[' '])
    leftleg_upper   =   rotation_estimation(joint['lhip'],joint['lkne'],transform['lhip'],transform['lkne'],mask[' '])
    rightleg_upper  =   rotation_estimation(joint['rhip'],joint['rkne'],transform['rhip'],transform['rkne'],mask[' '])
    leftleg_lower   =   rotation_estimation(joint['lkne'],joint['lank'],transform['lkne'],transform['lank'],mask[' '])
    rightleg_lower  =   rotation_estimation(joint['rkne'],joint['rank'],transform['rkne'],transform['rank'],mask[' '])
    head            =   rotation_estimation(joint['lear'],joint['rear'],transform['lear'],transform['rear'],mask[' '])
    body            =   rotation_estimation(joint['neck'],joint['pelv'],transform['neck'],transform['pelv'],mask[' '])

"""
  
