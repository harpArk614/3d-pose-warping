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

from google.colab import drive
drive.mount('/content/gdrive')

# !unzip '/content/gdrive/MyDrive/In-shop Clothes Retrieval Benchmark/Img/img.zip'

infile = open('/content/gdrive/MyDrive/poses_fashion3d.pkl','rb')
poses = pickle.load(infile)

poses['MEN']['Denim']

poses['MEN']['Denim']['id_00007216']['01']['7_additional']

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

img_to_tensor('img/MEN/Denim/id_00000080/01_1_front.jpg')

data = {}
for n in poses:
  for i in poses[n]:
  #data_men[i] = {} 
    for j in poses[n][i]:
    #data_men[i][j]={}
      for k in poses[n][i][j]:
      #data_men[i][j][k]={}
        for l in poses[n][i][j][k]:
          #data_men[i][j][k][l]={}
          #for m in poses[n][i][j][k][l]:
          #/img/MEN/Denim/id_00000182/01_1_front.jpg
          path = 'img/'+n+'/'+i+'/'+j+'/'+k+'_'+l+'.jpg'
          
          x = img_to_tensor(path)
          data.update({path : x})

#x = img_to_tensor(path)
#data_men[i][j][k][l]=x

# print(data["img/MEN/Denim/id_00000080/01_7_additional.jpg"])

# !mkdir Dataset

# !cd Dataset

tstImg2=np.round(np.array(Image.open('img/MEN/Denim/id_00000080/01_1_front.jpg')).convert('RGB').resize((224,224)),dtype=np.float32)

tf.reshape(tstImg2, shape=[-1, 224, 224, 3])

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

tensor2=tf.io.decode_image(
    '/content/img/MEN/Denim/id_00000080/01_1_front.jpg'
)

# img=Image.open('/content/img/MEN/Denim/id_00000080/01_1_front.jpg')
# array = tf.keras.preprocessing.image.img_to_array(img)

# print(array)

data = image.imread('/content/img/MEN/Denim/id_00000080/01_1_front.jpg')
plt.imshow(data)

# len(data_pose)

joint_order=['neck', 'nose', 'lsho', 'lelb', 'lwri', 'lhip', 'lkne', 'lank', 'rsho', 'relb', 'rwri', 'rhip', 'rkne', 'rank', 'leye', 'lear', 'reye', 'rear', 'pelv']

def give_name_to_keypoints(array, joint_order):
    #array = array.T
    res = {}
    for i, name in enumerate(joint_order):
        res[name] = array[i]
    return res

for i, name in enumerate(joint_order):
  print(i,name)

path="img/MEN/Denim/id_00000080/01_7_additional.jpg"
# print(data_pose.get(path))
print(data.get(path))

data_with_joints={}
for path,image in data.items():
  array=data.get(path)
  data_with_joints[path]=give_name_to_keypoints(array, joint_order)

data_with_joints["img/MEN/Denim/id_00000080/01_7_additional.jpg"]['lsho']

img_men = tf.keras.preprocessing.image_dataset_from_directory(
    'img/MEN',
    image_size=(256, 256),
    labels = 'inferred'
)

type(img_men)

img_men_training = tf.keras.preprocessing.image_dataset_from_directory(
    'img/MEN',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    labels = 'inferred'
)