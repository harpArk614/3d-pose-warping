def Discriminator():
  
  ini = layers.Input(shape=[256, 256, 3], name="input_img")
    
  
  t = Conv2D(kernel_size=3,
               strides=1,
               filters=64,
               padding="same")(ini)
  t = relu_bn(t)
    
  num_blocks_list = [2,5, 5,2]
  for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=64)
    
    
  output3 = t

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(output3) # (bs, 34, 34, 256)

  initializer = tf.random_normal_initializer(0., 0.02)
  conv = tf.keras.layers.Conv2D(512, 
                                kernel_size=4, 
                                strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
  norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 
                                kernel_size=4, 
                                strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
  last=tf.keras.layers.Flatten()(last)
  last=tf.keras.layers.Dense(128)(last)
  last=tf.keras.layers.Dense(64)(last) 
  last=tf.keras.layers.Dense(32)(last)  
  last=tf.keras.layers.Dense(1)(last)                                     
                                   

  return keras.Model(inputs=ini, outputs=last)
