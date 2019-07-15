# Modules
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, BatchNormalization, Flatten, PReLU, Activation, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

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


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, shape):
        self.image_shape =shape

    def generator(self):
        generator = Sequential()
        generator.add(Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same", input_shape=self.image_shape))
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
    
    gan.compile(loss={g:'mse', d: 'binary_crossentropy'},
            loss_weights=[1., 1e-3],
            optimizer='adam',
            metrics = {g:perceptual_distance, d: 'binary_crossentropy'}
            )
    return gan