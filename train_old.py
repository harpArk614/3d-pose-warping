cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)      
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
   
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator_g,
                                 discriminator=discriminator_y)
   
epochs=1
num_examples_to_generate = 16

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator_g(images, training=True)

      real_output = discriminator_y(images, training=True)
      fake_output = discriminator_y(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
     # print(gen_loss,disc_loss,real_output,fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_g.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_y.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_g.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_y.trainable_variables))
    
    
def train(datasetGen, epochs):
  for epoch in range(epochs):
    start = time.time()
 
    for image,labels in datasetGen:
      train_step(image)
 
     # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator_g,
                              epoch + 1,                                         
                              )
 
    # Save the model every 15 epochs
    if (epoch + 1) % 1 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
 
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
 
  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator_g,
                           epochs,
                           seed)
                           
                           
 train(datasetGen, epochs)     
