
import tensorflow as tf
from keras import layers
from tensorflow.keras import datasets, layers, models
from tensorflow import Tensor 
from tensorflow.keras.layers import Input, Conv3D, ReLU, Add, AveragePooling3D, Flatten, Dense
from tensorflow.keras.models import Model
#install tensorflow-addons
#from tensorflow_addons.layers import GroupNormalization

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = GroupNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv3D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv3D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = Conv3D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def pose_encoder():
    
    inputs = Input(shape=(32, 32, 64, 64))
    
    t = GroupNormalization()(inputs)
    t = Conv3D(kernel_size=3,
               strides=1,
               filters=64,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [1, 1, 1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=64)
    
    
    t = AveragePooling3D(4)(t)
    outputs = t
    model = Model(inputs, outputs)

    return model
