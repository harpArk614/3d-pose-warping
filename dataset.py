import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from matplotlib import image 
import glob
import os
from PIL import Image
from numpy import asarray
import PIL
import pathlib
import tensorflow_datasets as tfds
# import tensorflow.keras.datasets.cifar10 as cf


!unzip '/content/drive/MyDrive/DeepFashion/In-shop Clothes Retrieval Benchmark/Img/img.zip'

infile = open('/content/drive/MyDrive/poses_fashion3d.pkl','rb')
poses = pickle.load(infile)



#Function For Converting image into numpy array

def img_to_tensor (path):
  # load the image
  image = Image.open(path)
  # convert image to numpy array
  data = asarray(image)
  # print(type(data))
  # # summarize shape
  # print(data.shape)

  # # create Pillow image
  # image2 = Image.fromarray(data)
  # print(type(image2))

  # # summarize image details
  # print(image2.mode)
  # print(image2.size)
  data.reshape(256,256,3)
  return data



data = {}
for n in poses:
  for i in poses[n]:
  #data_men[i] = {} 
    for j in poses[n][i]:
    #data_men[i][j]={}
      for k in poses[n][i][j]:
      #data_men[i][j][k]={}
        for l in poses[n][i][j][k]:
          path = '/content/img/'+n+'/'+i+'/'+j+'/'+k+'_'+l+'.jpg'
          x = img_to_tensor(path)
          data.update({path : x})



data_pose={}
for n in poses:
  for i in poses[n]:
    for j in poses[n][i]:
      for k in poses[n][i][j]:
        for l in poses[n][i][j][k]:
          #for m in poses[n][i][j][k][l]:
          path = '/content/img/'+n+'/'+i+'/'+j+'/'+k+'_'+l+'.jpg'
          data_pose.update({path : poses[n][i][j][k][l]})



joint_order=['neck', 'nose', 'lsho', 'lelb', 'lwri', 'lhip', 'lkne', 'lank', 'rsho', 'relb', 'rwri', 'rhip', 'rkne', 'rank', 'leye', 'lear', 'reye', 'rear', 'pelv']

def give_name_to_keypoints(array, joint_order):
    res = {}
    for i, name in enumerate(joint_order):
        res[name] = array[i]
    return res


data_with_joints={}
for path,image in data.items():
  array=data_pose.get(path)
  data_with_joints[path]=give_name_to_keypoints(data_pose.get(path), joint_order)

