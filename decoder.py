# Function to create Dencoder model
#import encoder.py
#istall tensorflow-addons

def create_resnet_dencoder():
    inputen, outputen = create_resnet_encoder()
  

    t = outputen
    num_blocks_list = [2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block3d(t, downsample=0, filters=64)
    
    t = layers.Conv3D(10, (3, 3, 3), padding='same', strides=(1, 1,  1))(t)
    t = layers.Conv3D(2, (3, 3, 3), padding='same', strides=(1, 1,  1))(t)
    t = layers.Reshape((32, 32, 128))(t)
    t = layers.Conv2DTranspose(64,
                            kernel_size=4,
                            strides=2,
                            padding="same",
                            use_bias=False)(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=0, filters=64)
        t = layers.Conv2DTranspose(64,
                            kernel_size=4,
                            strides=2,
                            padding="same",
                            use_bias=False)(t)
        
    
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=128,
               padding="same")(t)
    t = relu_bn(t)

    t = layers.Conv2D(filters=64, kernel_size=3, padding='same')(t)
    t = layers.Conv2D(filters=32, kernel_size=3, padding='same')(t)
    t = layers.Conv2D(filters=16, kernel_size=3, padding='same')(t)
    t = layers.Conv2D(filters=3, kernel_size=3, padding='same')(t)
    outputs = t 
  
    model = Model(inputen, outputs)

    return model

