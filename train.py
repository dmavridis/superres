import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
# from wandb.keras import WandbCallback
# from tqdm import tqdm
from network import Generator, Discriminator, gan
from transforms import *

transforms_dict = {
                    0: flipx, 
                    1: flipy,
                    2: addbias,
                    3: rollx,
                    4: rolly,
                  }
keys = list(transforms_dict.keys())


run = wandb.init(project='superres')
config = run.config
config.num_epochs = 20#50
config.batch_size = 12
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

def image_generator(batch_size, img_dir):
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
        yield (small_images.astype('float32'), large_images.astype('float32'))
        counter += batch_size

def image_generator_aug(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= 2*len(input_filenames):
            counter = 0
        for i in range(batch_size):
            tr = np.random.randint(0,10)
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
            if tr in keys:
                small_images[i], large_images[i] = transforms_dict[tr](small_images[i], large_images[i])
        yield (small_images.astype('float32'), large_images.astype('float32'))
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

val_generator = image_generator(config.batch_size, val_dir)
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


input_shape = (config.input_width, config.input_height, 3)
output_shape = (config.output_width, config.output_height, 3)

g = Generator(input_shape).generator()
d = Discriminator(output_shape).discriminator()
srgan = gan(g, d, input_shape)


for e in range(1, config.num_epochs+1):
    for step in range(config.steps_per_epoch):

        image_batch_lr, image_batch_hr = next(image_generator_aug(config.batch_size, train_dir))
        generated_images_sr = g.predict(image_batch_lr)
        
        real_data_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
        fake_data_Y = np.random.random_sample(config.batch_size)*0.2
        
        d.trainable = True
        d_loss_real = d.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = d.train_on_batch(generated_images_sr, fake_data_Y)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
        image_batch_lr, image_batch_hr = next(image_generator_aug(config.batch_size, train_dir))
        
        gan_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
        d.trainable = False
        loss = srgan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        
    gan_loss, _, _, perc_dist, _ = loss
    wandb.log({
        'epoch': e,
        'loss': loss,
        'perceptual_distance': perc_dist})

# Test
    for _ in range(config.val_steps_per_epoch):
        image_batch_lr, image_batch_hr = next(image_generator(config.batch_size, val_dir))
        val_loss_gan = srgan.test_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        val_loss, _, _, val_perc_dist, _ = val_loss_gan
    wandb.log({
             'epoch': e,
             'val_loss': val_loss,
             'val_perceptual_distance': val_perc_dist})
    print(f'Epoch {e} => loss: {gan_loss:.2f}, perceptual distance: {perc_dist:.2f}, val_loss: {val_loss:.2f}, val_perceptual distance: {val_perc_dist:.2f}')
