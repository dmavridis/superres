import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, BatchNormalization, Flatten, PReLU, Activation, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model


num_epochs = 20#50
batch_size = 8
input_height = 32
input_width = 32
output_height = 256
output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // batch_size
val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // batch_size

##############  Network #######################################
        
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


# Residual block
def res_block_gen(model, kernel_size, filters, strides):
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", use_bias=False))
    model.add(BatchNormalization(momentum = 0.5))
    # Using Parametric ReLU
    model.add(PReLU(shared_axes=[1,2]))
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", use_bias=False))
    model.add(BatchNormalization(momentum = 0.5))
    return model
  
    
def up_sampling_block(model, kernel_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same"))
    model.add(UpSampling2D(size = 2))
    model.add(LeakyReLU(alpha = 0.2))
    return model

def discriminator_block(model, filters, kernel_size, strides):
    model.add(Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", use_bias=False))
    model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(alpha = 0.2))
    return model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, shape):
        self.image_shape =shape

    def generator(self):
        generator = Sequential()
        generator.add(Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same", input_shape=self.image_shape))
        generator.add(PReLU(shared_axes=[1,2]))
        # Using 16 Residual Blocks
        for _ in range(16):
            generator = res_block_gen(generator, 3, 64, 1)
        generator.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", use_bias=False))
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

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, shape):
        self.image_shape = shape
    
    def discriminator(self):
        discriminator = Sequential()
        discriminator.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", input_shape=self.image_shape))
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
        discriminator.compile(optimizer=opt, loss='mse', metrics=['acc'])
        return discriminator
            
def gan(generator, discriminator, shape):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    generated_images = generator(gan_input)
    outputs = discriminator(generated_images)
    gan = Model(inputs=gan_input, outputs=[generated_images, outputs])
    g = gan.get_layer(index=1).name
    d = gan.get_layer(index=2).name
    
    gan.compile(loss={g:perceptual_distance, d: 'binary_crossentropy'},
            loss_weights=[1., 1e-3],
            optimizer='adam',
            metrics = {g:perceptual_distance, d: 'binary_crossentropy'}
            )
    return gan


######## Transforms #################
    

def flipx(image_lr, image_hr):
    return np.flip(image_lr,axis=0), np.flip(image_hr,axis=0)

def flipy(image_lr, image_hr):
    return np.flip(image_lr,axis=1), np.flip(image_hr,axis=1)

def addbias(image_lr, image_hr):
    bias = np.random.normal(0.1,0.1)
    return np.clip(image_lr + bias, 0, 1), np.clip(image_hr + bias, 0, 1)

def rollx(image_lr, image_hr):
    shift = np.random.randint(0,5)
    return np.roll(image_lr, shift, axis=0), np.roll(image_hr, 8*shift, axis=0)

def rolly(image_lr, image_hr):
    shift = np.random.randint(0,5)
    return np.roll(image_lr, shift, axis=1), np.roll(image_hr, 8*shift, axis=1)    
        
transforms_dict = {
                    0: flipx, 
                    1: flipy,
                    2: addbias,
                    3: rollx,
                    4: rolly,
                  }
keys = list(transforms_dict.keys())        

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
            (batch_size, input_width, input_height, 3))
        large_images = np.zeros(
            (batch_size, output_width, output_height, 3))
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
            (batch_size, input_width, input_height, 3))
        large_images = np.zeros(
            (batch_size, output_width, output_height, 3))
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


val_generator = image_generator(batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)

##############  Preprocessing ####################

input_shape = (input_width, input_height, 3)
output_shape = (output_width, output_height, 3)

g = Generator(input_shape).generator()
d = Discriminator(output_shape).discriminator()
srgan = gan(g, d, input_shape)

################ Train #########################
for e in range(1, num_epochs+1):
    for step in range(steps_per_epoch):

        image_batch_lr, image_batch_hr = next(image_generator_aug(batch_size, train_dir))
        generated_images_sr = g.predict_on_batch(image_batch_lr)
        
        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        fake_data_Y = np.random.random_sample(batch_size)*0.2
        
        d.trainable = True
        d_loss_real = d.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = d.train_on_batch(generated_images_sr, fake_data_Y)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
        image_batch_lr, image_batch_hr = next(image_generator_aug(batch_size, train_dir))
        
        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        d.trainable = False
        loss = srgan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        
    gan_loss, _, _, perc_dist, _ = loss

#################### Test ##########################
    for _ in range(val_steps_per_epoch):
        image_batch_lr, image_batch_hr = next(image_generator(batch_size, val_dir))
        val_loss_gan = srgan.test_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        val_loss, _, _, val_perc_dist, _ = val_loss_gan

    print(f'Epoch {e} => loss: {gan_loss:.2f}, perceptual distance: {perc_dist:.2f}, val_loss: {val_loss:.2f}, val_perceptual distance: {val_perc_dist:.2f}')
