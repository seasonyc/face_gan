# -*- coding: utf-8 -*-

from __future__ import print_function, division


import keras.backend as K

import matplotlib.pyplot as plt

import sys

import tensorflow as tf
from tensorflow import keras

import time
import datetime
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, ZeroPadding2D, Activation, Add, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Concatenate, RepeatVector, Reshape, Lambda
from tensorflow.keras.models import Model
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.losses import mae, binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.regularizers import l2
import dataset
from utils import save_model, load_model, img_renorm, plot_image, plot_image_list, read_image

lambda_gp = 10
lambda_rec = 5 #starGAN 10, attGAN 100
batch_size = 32
lr_decay_ratio = 0.8
_epochs=10


    

class FaceGAN():
    def __init__(self, learning_rate, generator_norm_func = InstanceNormalization, discriminator_norm_func = None, weight_decay=1e-4):
        from tensorflow.keras.models import model_from_json
     
        #facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
        self.facenet = model_from_json(open("model/facenet_model.json", "r").read())
        #pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
        self.facenet.load_weights('model/facenet_weights.h5')
        self.facenet.trainable = False
        
        self.generator_norm_func = generator_norm_func
        self.discriminator_norm_func = discriminator_norm_func
        self.weight_decay = weight_decay
        # Shape of images
        self.image_shape = (160, 160, 3)
        
        
        self.n_critic = 5
        
        # it's rational to set lr in train by LearningRateScheduler, but we don't require such dynamic
        self.build_model(learning_rate)
        


    
    def build_model(self, learning_rate):
        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        #-------------------------------
        # Construct Computational Graph
        #       for the discriminator
        #-------------------------------

        # Freeze generator's layers while training discriminator
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.image_shape)

        fake_img = self.generator(real_img)

        # Discriminator determines validity of the real and fake images
        d_out_fake = self.discriminator(fake_img)
        d_out_real = self.discriminator(real_img)


        # Construct weighted average between real and fake images
        def random_interpolate(inputs):
            alpha = K.random_uniform((batch_size, 1, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
        interpolated_img = Lambda(random_interpolate)([real_img, fake_img])
        # Determine validity of weighted sample
        d_out_interpolated = self.discriminator(interpolated_img)

        def gradient_penalty_loss():
            """
            Computes gradient penalty based on prediction and weighted real / fake samples
            """
            gradients = K.gradients(d_out_interpolated, [interpolated_img])[0]
            gradient_l2_norm = K.sqrt(K.sum(K.batch_flatten(K.square(gradients)), axis=-1))
            
            return K.mean(K.square(1 - gradient_l2_norm))
    

        wasserstein_loss = K.mean(d_out_fake) - K.mean(d_out_real)
        
        discriminator_loss = wasserstein_loss + lambda_gp * gradient_penalty_loss()

        self.discriminator_training_model = Model(real_img, [d_out_fake, d_out_real, d_out_interpolated])
        self.discriminator_training_model.add_loss(discriminator_loss)
        d_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-08)
        self.discriminator_training_model.compile(optimizer=d_optimizer)
        self.discriminator_training_model.summary()



        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        input_img = Input(shape=self.image_shape)

        gen_img = self.generator(input_img)
        d_out = self.discriminator(gen_img)
        # Defines generator model
        # It's ok to output gen_img 
        # then don't need to set discriminator.trainable = False because then discriminator is not a part of model to be trained.
        self.generator_training_model = Model(input_img, d_out)
        
        def perceptual_loss(pm, selected_pm_layers, selected_pm_weights, input_img, rec_img):
            '''Perceptual loss for the DFC VAE'''
            outputs = [pm.get_layer(l).output for l in selected_pm_layers]
            
            model = Model(pm.input, outputs)
        
            h1_list = model(input_img)
            h2_list = model(rec_img)
            weights = selected_pm_weights
            if not isinstance(h1_list, list):
                h1_list = [h1_list]
                h2_list = [h2_list]
                weights = [weights]
                    
            p_loss = 0.0
            
            for h1, h2, weight in zip(h1_list, h2_list, weights):
                h1 = K.batch_flatten(h1)
                h2 = K.batch_flatten(h2)
                p_loss = p_loss + weight * K.mean(K.abs(h1 - h2), axis=-1)
            
            return p_loss


        generator_rec_loss = K.mean(perceptual_loss(self.facenet, ['Conv2d_1a_3x3', 'Conv2d_2b_3x3', 'Conv2d_4a_3x3', 'Conv2d_4b_3x3', 'Bottleneck'], [1, 1, 1, 1, 1], input_img, gen_img))
        generator_wasserstein_loss = -K.mean(d_out)
        generator_loss = generator_wasserstein_loss + lambda_rec * generator_rec_loss
        
        self.generator_training_model.add_loss(generator_loss)
        g_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-08)
        self.generator_training_model.compile(optimizer=g_optimizer)
        self.generator_training_model.summary()
        
        
    def build_generator(self):
        image = Input(shape=self.image_shape, name='input_image')
        
        code = self.facenet(image)
        
        x = Dense(1792, use_bias=False)(code)
        x = self.generator_norm_func()(x)
        x = LeakyReLU(alpha=0.01)(x) #alpha=0.2
        x = Reshape((1, 1, 1792))(x)
        
        channels = 1024
        for i in range(2):
            x = self.deconv_block(x, channels, padding = 'valid', kernel_size = 3, strides = 1)
            print(K.int_shape(x))
            channels //= 2
            
        for i in range(4):
            x = self.deconv_block(x, channels)
            print(K.int_shape(x))
            channels //= 2
        
        x = Conv2DTranspose(3, 4, padding='same', strides=2)(x)
        print(K.int_shape(x))
        x = Activation('tanh', name='gen_image')(x) #tanh to ensure output is between -1, 1
            
        
        return Model(image, x, name='generator')



    def build_discriminator(self):
        input = Input(shape=self.image_shape, name='image')
        x = input
        channels = 32
        repeat = 5
        for i in range(repeat):
            channels *= 2
            x = self.downsampling_conv_block(x, channels)
            print(K.int_shape(x))
          
        x = Flatten()(x)
        
        x_src = Dense(1024, use_bias=False)(x)
        if self.discriminator_norm_func:
            x_src = self.discriminator_norm_func()(x_src)
        x_src = LeakyReLU(alpha=0.01)(x_src)
        x_src = Dense(1)(x_src)
        '''
        x_cls = Dense(1024, use_bias=False)(x)
        x_cls = layer_normalization(x_cls)
        x_cls = LeakyReLU(alpha=0.01)(x_cls)
        x_cls = Dense(1, activation='sigmoid')(x_cls)
        '''
        return Model(input, x_src, name='discriminator')

    
    
    def downsampling_conv_block(self, x, channels, kernel_size = 4, strides = 2):
        x = ZeroPadding2D()(x)
        x = Conv2D(channels, kernel_size, strides=strides, use_bias=False)(x)
        if self.discriminator_norm_func:
            x = self.discriminator_norm_func()(x)
        x = LeakyReLU(alpha=0.01)(x) 
        return x
    
      
    
    def deconv_block(self, x, channels, padding = 'same', kernel_size = 4, strides = 2):
        x = Conv2DTranspose(channels, kernel_size, padding=padding, strides=strides, use_bias=False)(x)
        x = self.generator_norm_func()(x)
        x = LeakyReLU(alpha=0.01)(x) #alpha=0.2
        return x
    

    def train(self, epochs):

        x_train, train_size = dataset.load_celeba('CelebA', batch_size, part='train', consumer = 'translator')
        x_val, val_size = dataset.load_celeba('CelebA', batch_size, part='val', consumer = 'translator')

        x_train_itr = x_train.make_one_shot_iterator()
        x_train_next = x_train_itr.get_next()
        x_val_itr = x_val.make_one_shot_iterator()
        x_val_next = x_val_itr.get_next()

        steps_per_epoch = train_size // batch_size
        validation_steps = val_size // batch_size
        sess = K.get_session()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for step in range(steps_per_epoch):
                train_img = sess.run(x_train_next)
                
                # Train discriminator
                d_loss = self.discriminator_training_model.train_on_batch(train_img)
                
                
                # Train Generator
                if (step+1) % self.n_critic == 0:
                    g_loss = self.generator_training_model.train_on_batch(train_img)

                    # Btw, print log...
                    et = time.time() - epoch_start_time
                    eta = et * (steps_per_epoch - step - 1) / (step + 1) 
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    eta = str(datetime.timedelta(seconds=eta))[:-7]
                    log = "{}/{}   - Elapsed: {}, ETA: {}  - d_loss: {:.4f} , g_loss: {:.4f}".format(step+1, steps_per_epoch, et, eta, d_loss, g_loss)
                    print(log)
            # validate per epoch
            d_val_loss = 0
            g_val_loss = 0
            img_rec_acc = 0
            for step in range(validation_steps):
                val_img = sess.run(x_val_next)
                d_val_loss += self.discriminator_training_model.test_on_batch(val_img)
                g_val_loss += self.generator_training_model.test_on_batch(val_img)
                # rec_img = self.generator.predict_on_batch(val_img)
                # img_rec_acc += K.mean(1 - mae(K.batch_flatten(val_img), K.batch_flatten(rec_img)) / 2)
            
            d_val_loss /= validation_steps
            g_val_loss /= validation_steps
            img_rec_acc /= validation_steps     

            
            log = "ephoch {}   - d_val_loss: {:.4f} , g_val_loss: {:.4f} , img_rec_acc: {:.4f}  - d_loss: {:.4f} , g_loss: {:.4f}".format(epoch+1, d_val_loss, g_val_loss, img_rec_acc, d_loss, g_loss)
            print(log)
            
            # save model per epoch
            save_model(self.generator, 'face_gan_epoch{:02d}-d_loss{:.4f}-g_loss{:.4f}-acc{:.4f}'.format(epoch+1, d_val_loss, g_val_loss, img_rec_acc) )
            
            # test the model
            self.test_gan(epoch)
            
            # update learning rate
            lr = K.get_value(self.discriminator_training_model.optimizer.lr)
            lr *= lr_decay_ratio
            K.set_value(self.discriminator_training_model.optimizer.lr, lr)
            K.set_value(self.generator_training_model.optimizer.lr, lr)
            lr = K.get_value(self.discriminator_training_model.optimizer.lr)
            print(str(lr))
            lr = K.get_value(self.generator_training_model.optimizer.lr)
            print(str(lr))
        
    def test_gan(self, epoch):
        for part in ('train', 'val', 'test'):
            images = dataset.fetch_smallbatch_from_celeba('CelebA', part=part)
            rec_imgs = self.generator.predict(images)
            plot_image(img_renorm(images), img_renorm(rec_imgs), epoch = epoch)


gan = FaceGAN(learning_rate = 0.0001, generator_norm_func = InstanceNormalization, discriminator_norm_func = InstanceNormalization)
tf.logging.set_verbosity(tf.logging.ERROR)
gan.train(epochs=_epochs)
