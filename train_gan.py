import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 2#50
config.batch_size = 16
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

#config.steps_per_epoch = len(
#    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
#config.val_steps_per_epoch = len(
#    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size
    
config.steps_per_epoch = 2
config.val_steps_per_epoch = 2

def image_generator_d(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size

def image_generator_g(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size




def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

val_generator = image_generator_g(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)

class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)


from network import Generator, Discriminator, get_gan_network

shape = (config.input_width, config.input_height, 3)
image_shape = (config.output_width, config.output_height, 3)

generator = Generator(shape).generator()
discriminator = Discriminator(image_shape).discriminator()

#adam = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator.compile(loss="mse", optimizer="adam")
discriminator.compile(loss="binary_crossentropy", optimizer="adam")

gan = get_gan_network(discriminator, shape, generator, "adam", gan_metric=perceptual_distance )

for e in range(1, config.num_epochs+1):
    print ('-'*15, 'Epoch %d' % e, '-'*15)
    for _ in range(config.steps_per_epoch):
        
        image_batch_lr, image_batch_hr = next(image_generator_g(config.batch_size, train_dir))
        generated_images_sr = generator.predict(image_batch_lr)

        real_data_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
        fake_data_Y = np.random.random_sample(config.batch_size)*0.2
        
        discriminator.trainable = True
        
        d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
        #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
        image_batch_lr, image_batch_hr = next(image_generator_d(config.batch_size, train_dir))
        
        gan_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
        discriminator.trainable = False
        loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
    print("Loss HR , Loss LR, Loss GAN", "Perceptual Distance")
    print(d_loss_real, d_loss_fake, loss_gan)


#        if e == 1 or e % 5 == 0:
#            plot_generated_images(e, generator)
#        if e % 300 == 0:
#            generator.save('./output/gen_model%d.h5' % e)
#            discriminator.save('./output/dis_model%d.h5' % e)
#            gan.save('./output/gan_model%d.h5' % e)

#    
#    
## DONT ALTER metrics=[perceptual_distance]
#model.compile(optimizer='adam', loss='mse',
#              metrics=[perceptual_distance])
#
#model.fit_generator(image_generator(config.batch_size, train_dir),
#                    steps_per_epoch=config.steps_per_epoch,
#                    epochs=config.num_epochs, callbacks=[
#                        ImageLogger(), WandbCallback()],
#                    validation_steps=config.val_steps_per_epoch,
#                    validation_data=val_generator)