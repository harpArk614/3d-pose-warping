#Function for 2D Residual Block

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def residual_block_decode(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv2DTranspose(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
  
    if downsample:
        x = layers.Conv2DTranspose(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


#Function for 3D Residual Block

def relu_bn3d(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = GroupNormalization()(relu)
    return bn

def residual_block3d(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv3D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn3d(y)
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
    out = relu_bn3d(out)
    return out


# Function to create Encoder model

def create_resnet_encoder():
    
    inputs = Input(shape=(256, 256, 3))
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=64,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 2, 3, 3]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=64)
    
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=128,
               padding="same")(t)
    t = layers.Reshape((32, 32, 64, 2))(t)
    t = layers.Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1,  1))(t)

    num_blocks_list = [2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block3d(t, downsample=(j==0 and i!=0), filters=64)

    outputs = t 
  
    return (inputs,outputs)

