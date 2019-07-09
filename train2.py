#!/usr/bin/env python

import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, LambdaCallback
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, BatchNormalization, Flatten, PReLU, Activation, UpSampling2D

from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from tqdm import tqdm


run = wandb.init(project='superres')
config = run.config

config.num_epochs = 4#50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
config.channels = 3
config.input_shape = (config.input_height, config.input_width, config.channels)
config.output_shape = (config.output_height, config.output_width, config.channels)



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
    
config.steps_per_epoch = 20
config.val_steps_per_epoch = 20


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
        yield (small_images, large_images)
        counter += batch_size

def gan_image_generator(image_generator, img_dir):
        image_batch_lr, _ = next(image_generator(config.batch_size, train_dir))
        gan_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
        yield image_batch_lr, gan_Y

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


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


def log_discriminator(epoch, logs):
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

    
# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    model.add(Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same"))
    model.add(BatchNormalization(momentum = 0.5))
    # Using Parametric ReLU
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]))
    model.add(Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same"))
    model.add(BatchNormalization(momentum = 0.5))
    return model

def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model.add(Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same"))
    model.add(UpSampling2D(size = 2))
    model.add(LeakyReLU(alpha = 0.2))
    return model

def discriminator_block(model, filters, kernel_size, strides):
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha = 0.2))
    return model
    
def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", input_shape=config.output_shape))
    discriminator.add(LeakyReLU(alpha = 0.2))
    discriminator = discriminator_block(discriminator, 32, 3, 2)
    discriminator = discriminator_block(discriminator, 64, 3, 1)
    discriminator = discriminator_block(discriminator, 64, 3, 2)
    discriminator = discriminator_block(discriminator, 128, 3, 1)
    discriminator = discriminator_block(discriminator, 128, 3, 2)
    discriminator = discriminator_block(discriminator, 256, 3, 1)
    discriminator = discriminator_block(discriminator, 256, 3, 2)
    discriminator.add(Flatten())
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha = 0.2))
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    discriminator.summary()
    
    opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(optimizer=opt, loss=wasserstein_loss, metrics=['acc'])
    return discriminator

def create_generator():
    generator = Sequential()
    generator.add(Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same", input_shape=config.input_shape))
    generator.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]))
    # Using 16 Residual Blocks
    for _ in range(16):
        generator = res_block_gen(generator, 3, 64, 1)
    generator.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same"))
    generator.add(BatchNormalization(momentum = 0.5))
    # Using 2 UpSampling Blocks
    for _ in range(3):
        generator = up_sampling_block(generator, 3, 256, 1)
    generator.add(Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same"))
    generator.add(Activation('relu'))
    generator.summary()
    
    opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(optimizer=opt, loss=wasserstein_loss, metrics=[perceptual_distance])
    return generator
    
def gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return gan
    
def mix_data(image_dir):
    image_batch_lr, image_batch_hr = next(image_generator(config.batch_size, image_dir))
    generated_images_sr = generator.predict(image_batch_lr)
    real_data_Y = np.ones(config.batch_size) - np.random.random_sample(config.batch_size)*0.2
    fake_data_Y = np.random.random_sample(config.batch_size)*0.2
    combined = np.concatenate((image_batch_hr, generated_images_sr),axis=0)
    labels = np.concatenate((real_data_Y, fake_data_Y),axis=0)
    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined_hr_inputs = combined[indices]
    labels = labels[indices] 
    yield combined_hr_inputs, labels


def train_discriminator(generator, discriminator):
#    train_inputs, train_labels = mix_data(train_dir)
#    val_inputs, val_labels = mix_data(generator, val_dir)
    discriminator.trainable = True
#    discriminator.summary()
    
    wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)

    discriminator.fit_generator(mix_data(train_dir), epochs=1,
                                steps_per_epoch=config.steps_per_epoch,
                                validation_steps=config.val_steps_per_epoch,
                                validation_data=val_generator,
                                callbacks = [wandb_logging_callback]
                                )
#    discriminator.save(path.join(wandb.run.dir, "discriminator.h5"))
    
def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'perceptual_distance': logs['perceptual_distance'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})


def train_generator(generator, discriminator, gan):
    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)
    discriminator.trainable = False
    gan.fit_generator(gan_image_generator, epochs=1,
                      steps_per_epoch=config.steps_per_epoch,
                      callbacks=[wandb_logging_callback])

discriminator = create_discriminator()
generator = create_generator()
srgan = gan(generator, discriminator)

for epoch in range(config.num_epochs):
    print(epoch)
    train_discriminator(generator, discriminator)
    train_generator(generator, discriminator, srgan)
