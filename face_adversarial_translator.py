# -*- coding: utf-8 -*-


import sys

import tensorflow as tf
from tensorflow import keras

import time
import datetime
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, ZeroPadding2D, Activation, Add, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.layers import Concatenate, RepeatVector, Reshape, Lambda
from tensorflow.keras.models import Model
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.losses import binary_crossentropy, mae
from tensorflow.keras.regularizers import l2
import dataset
from utils import save_model, load_model, img_renorm, plot_image, plot_image_list, read_image, plot_images

lambda_gp = 10
lambda_rec = 10 #starGAN 10, attGAN 100
lambda_cls_gen = 1 #starGAN 1, attGAN 10
lambda_cls_real = 1 #starGAN 1, attGAN 1

batch_size = 32
lr_decay_ratio = 0.95
_epochs=10


   

class FaceGAN():
    def __init__(self, learning_rate, generator_norm_func = InstanceNormalization, discriminator_norm_func = None, weight_decay=1e-4):
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
        label = Input(shape=(1,))
        fake_label = Input(shape=(1,))

        fake_img = self.generator([real_img, fake_label])

        # Discriminator determines validity of the real and fake images
        src_fake, _, _ = self.discriminator(fake_img)
        src_real, _, cls_real = self.discriminator(real_img)


        # Construct weighted average between real and fake images
        def random_interpolate(inputs):
            alpha = K.random_uniform((batch_size, 1, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
        interpolated_img = Lambda(random_interpolate)([real_img, fake_img])
        # Determine validity of weighted sample
        src_interpolated, _, _ = self.discriminator(interpolated_img)

        '''
        discriminator_loss = wasserstein_loss + lambda_gp * gradient_penalty_loss() + lambda_cls_real * real_cls_loss
        self.discriminator_training_model.add_loss([wasserstein_loss, lambda_gp * gradient_penalty_loss(), lambda_cls_real * real_cls_loss])
        to track the parts of the loss, use the ugly way...
        '''
        def gradient_penalty_loss(dummy_true, dummy_pre):
            """
            Computes gradient penalty based on prediction and weighted real / fake samples
            """
            gradients = K.gradients(src_interpolated, [interpolated_img])[0]
            gradient_l2_norm = K.sqrt(K.sum(K.batch_flatten(K.square(gradients)), axis=-1))
            
            return K.mean(K.square(1 - gradient_l2_norm))
    
        def wasserstein_loss(dummy_true, dummy_pre):
            return K.mean(src_fake) - K.mean(src_real)
            
        def real_cls_loss(dummy_true, dummy_pre):
            return K.mean(binary_crossentropy(label, cls_real))
        
        self.discriminator_training_model = Model([real_img, label, fake_label],  [src_fake, cls_real, src_interpolated])#[src_fake, src_real, cls_real, src_interpolated])
        d_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-08)
        self.discriminator_training_model.compile(optimizer=d_optimizer, loss=[wasserstein_loss, gradient_penalty_loss, real_cls_loss], loss_weights=[1, lambda_gp, lambda_cls_real])
        self.discriminator_training_model.summary()



        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        input_img = Input(shape=self.image_shape)
        orig_label = Input(shape=(1,))
        target_label = Input(shape=(1,))

        gen_img = self.generator([input_img, target_label])
        src_gen, cls_amp_gen, cls_gen = self.discriminator(gen_img)
        rec_img = self.generator([input_img, orig_label])
        # Defines generator model
        # It's ok to output gen_img 
        # then don't need to set discriminator.trainable = False because then discriminator is not a part of model to be trained.
        self.generator_training_model = Model([input_img, orig_label, target_label], [rec_img, src_gen, cls_gen])
        '''
        from tensorflow.keras.models import model_from_json
     
        #facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
        facenet = model_from_json(open("model/facenet_model.json", "r").read())
        #pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
        facenet.load_weights('model/facenet_weights.h5')
        facenet.trainable = False


        def perceptual_loss(pm, selected_pm_layers, selected_pm_weights, input_img, rec_img):
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


        '''
        def rec_loss(dummy_true, dummy_pre):
            return K.mean(mae(K.batch_flatten(input_img), K.batch_flatten(rec_img)))
        
        def generator_wasserstein_loss(dummy_true, dummy_pre):
            return  -K.mean(src_gen)
        
        '''
        classifier doesn't requires Lipschitz constraint, however gp loss make
        gradient prior to cls dense not too much. if gen_class_loss impact
        too much, will add Lipschitz constraint on cls_amplitude
        '''
        def gen_class_loss(label, cls_amp_gen):
            label = label * 2 - 1
            return -K.mean(label * cls_amp_gen)
        
        def generater_cls_loss(dummy_true, dummy_pre):
            return  K.mean(binary_crossentropy(target_label, cls_gen))
         
        '''
        generator_loss = generator_wasserstein_loss + lambda_rec * generator_rec_loss + lambda_cls_gen * gen_class_loss(target_label, cls_amp_gen)
        self.generator_training_model.add_loss(generator_loss)
        '''
        
        g_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, epsilon=1e-08)
        self.generator_training_model.compile(optimizer=g_optimizer, loss=[generator_wasserstein_loss, rec_loss, generater_cls_loss], loss_weights=[1, lambda_rec, lambda_cls_gen])
        self.generator_training_model.summary()
        
        
    def build_generator(self):
        image = Input(shape=self.image_shape, name='input_image')
        
        label = Input(shape=(1,), name='target_label')
        
        def reshape_labels(size):
            def func(labels):
                labels = RepeatVector(size * size)(labels)
                labels = Reshape((size, size, 1))(labels)
                return labels
            return Lambda(func)

        reshaped_label = reshape_labels(self.image_shape[0])(label)
        image_label = Concatenate()([image,reshaped_label])
        
        channels = 32
        x = Conv2D(channels, 7, padding='same', use_bias=False, kernel_initializer='glorot_normal')(image_label)
        print(K.int_shape(x))
        
        # downsample
        for i in range(2):
            channels *= 2
            x = self.downsampling_conv_block(x, channels, norm_func=self.generator_norm_func, activation = ReLU())
            print(K.int_shape(x))
        
        # residul
        for i in range(5):
            x = self.res_block(x, channels, norm_func=self.generator_norm_func)
            print(K.int_shape(x))
        
        # upsample
        for i in range(2):
            channels //= 2
            x = self.deconv_block(x, channels, norm_func=self.generator_norm_func)
            print(K.int_shape(x))
        
        x = self.conv_block(x, 3, norm_func=self.generator_norm_func, kernel_size = 7)
        print(K.int_shape(x))
        
        x = Activation('tanh', name='gen_image')(x) #tanh to ensure output is between -1, 1
            
        
        return Model([image, label], x, name='generator')



    def build_discriminator(self):
        input = Input(shape=self.image_shape, name='image')
        channels = 64
        x = ZeroPadding2D()(input)
        x = Conv2D(channels, 4, strides=(2, 2), use_bias=False, kernel_initializer='glorot_normal')(x)
        print(K.int_shape(x))
        repeat = 4
        for i in range(repeat):
            channels *= 2
            x = self.downsampling_conv_block(x, channels, norm_func=self.discriminator_norm_func)
            print(K.int_shape(x))
          
        if self.discriminator_norm_func:
            x = self.discriminator_norm_func()(x)
        x = LeakyReLU(alpha=0.01)(x) #alpha=0.2
        
        x = Flatten()(x)
        
        x_src = Dense(1024, use_bias=False, kernel_initializer='glorot_normal')(x)
        if self.discriminator_norm_func:
            x_src = self.discriminator_norm_func()(x_src)
        x_src = LeakyReLU(alpha=0.01)(x_src)
        x_src = Dense(1, kernel_initializer='glorot_normal')(x_src)
        
        x_cls = Dense(1024, use_bias=False, kernel_initializer='glorot_normal')(x)
        if self.discriminator_norm_func:
            x_cls = self.discriminator_norm_func()(x_cls)
        x_cls = LeakyReLU(alpha=0.01)(x_cls)
        x_cls = Dense(1, name='cls_amplitude', kernel_initializer='glorot_normal')(x_cls)
        x_cls_sig = Activation('sigmoid', name='cls')(x_cls)
        
        return Model(input, [x_src, x_cls, x_cls_sig], name='discriminator')

    def conv_block(self, x, channels, norm_func, kernel_size = 4):
        if norm_func:
            x = norm_func()(x)
        x = ReLU()(x) #alpha=0.2
        x = Conv2D(channels, kernel_size, padding='same', use_bias=False, kernel_initializer='glorot_normal')(x)
        return x
    
    def downsampling_conv_block(self, x, channels, norm_func, kernel_size = 4, activation = LeakyReLU(alpha=0.01)):
        if norm_func:
            x = norm_func()(x)
        x = activation(x) #alpha=0.2
        x = ZeroPadding2D()(x)
        x = Conv2D(channels, kernel_size, strides=(2, 2), use_bias=False, kernel_initializer='glorot_normal')(x)
        return x
    
    
    def res_block(self, x, channels, norm_func, kernel_size = 3):
        input_x = x
        x = self.conv_block(x, channels, norm_func=norm_func, kernel_size = kernel_size)
        x = self.conv_block(x, channels, norm_func=norm_func, kernel_size = kernel_size)
        x = Add()([input_x, x])
        return x

    
    def upsampling_conv_block(self, x, channels, norm_func, kernel_size = 3):
        x = UpSampling2D()(x)
        x = self.conv_block(x, channels, norm_func=norm_func, kernel_size = kernel_size)
        return x
      
    
    def deconv_block(self, x, channels, norm_func, padding = 'same', kernel_size = 4):
        if norm_func:
            x = norm_func()(x)
        x = ReLU()(x) 
        x = Conv2DTranspose(channels, kernel_size, padding=padding, strides=(2, 2), use_bias=False, kernel_initializer='glorot_normal')(x)
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
        
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))
        
        def binary_accuracy(y_true, y_pre):
            return np.mean(np.fabs(y_true - y_pre) < 0.5) 

        for epoch in range(epochs):
            epoch_start_time = time.time()
            d_loss = np.array([0., 0., 0., 0.])
            g_loss = np.array([0., 0., 0., 0.])
            for step in range(steps_per_epoch):
                train_img, train_label = sess.run(x_train_next)
                train_target_label = 1 - train_label
                
                # Train discriminator
                d_loss += self.discriminator_training_model.train_on_batch([train_img, train_label, train_target_label])
                
                # Train Generator
                if (step+1) % self.n_critic == 0:
                    g_loss += self.generator_training_model.train_on_batch([train_img, train_label, train_target_label])


                # print log...
                if (step+1) % 100 == 0:#(step+1) % 100 == 0:
                    et = time.time() - epoch_start_time
                    eta = et * (steps_per_epoch - step - 1) / (step + 1) 
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    eta = str(datetime.timedelta(seconds=eta))[:-7]
                    
                    '''
                    stat latest 100 itrs of losses, to make the data not too old and not too variant
                    '''
                    d_loss /= 100
                    g_loss /= (100 / self.n_critic)
                    log = "{}/{}   - Elapsed: {}, ETA: {}  - d_loss: {:.4f} , w {:.4f} , gp {:.4f} , cls {:.4f}   , g_loss: {:.4f} , w {:.4f} , rec {:.4f} , cls {:.4f}"\
                        .format(step+1, steps_per_epoch, et, eta, d_loss[0], d_loss[1], d_loss[2], d_loss[3], g_loss[0], g_loss[1], g_loss[2], g_loss[3])
                    print(log)
                    d_loss.fill(0.)
                    g_loss.fill(0.)
                    
            # validate per epoch
            '''
            g: img_acc, gen_img_cls_acc
            d: real_validate, fake_validate, real_cls_acc
            use generator and discriminator, compose to get above metrics
            '''
            
            d_val_loss = np.array([0., 0., 0., 0.])
            g_val_loss = np.array([0., 0., 0., 0.])
            
            img_acc = 0
            gen_cls_acc = 0
            real_validate_acc = 0
            fake_validate_acc = 0
            real_cls_acc = 0
            for step in range(validation_steps):
                val_img, val_label = sess.run(x_val_next)
                val_target_label = 1 - val_label
                d_val_loss += self.discriminator_training_model.test_on_batch([val_img, val_label, val_target_label])
                g_val_loss += self.generator_training_model.test_on_batch([val_img, val_label, val_target_label])
                
                rec_img = self.generator.predict_on_batch([val_img, val_target_label])
                r_src, _, r_cls = self.discriminator.predict_on_batch(val_img)
                g_src, _, g_cls = self.discriminator.predict_on_batch(rec_img)
                rec_img = self.generator.predict_on_batch([val_img, val_label])
                
                real_validate_acc += binary_accuracy(np.ones((batch_size, 1), np.float32), sigmoid(r_src))
                fake_validate_acc += binary_accuracy(np.zeros((batch_size, 1), np.float32), sigmoid(g_src))
                real_cls_acc += binary_accuracy(val_label, r_cls.flatten())
                gen_cls_acc += binary_accuracy(val_target_label, g_cls.flatten())
                img_acc += (1 - np.mean(np.fabs(val_img.flatten() - rec_img.flatten())) / 2)
                
             
            d_val_loss /= validation_steps
            g_val_loss /= validation_steps
            
            img_acc /= validation_steps
            gen_cls_acc /= validation_steps
            real_validate_acc /= validation_steps
            fake_validate_acc /= validation_steps
            real_cls_acc /= validation_steps

            
            log = 'ephoch {}  validation:  - d_loss: {:.4f} , w {:.4f} , gp {:.4f} , cls {:.4f}   , '\
                        'g_loss: {:.4f} , w {:.4f} , rec {:.4f} , cls {:.4f}  -  '\
                        'img_acc: {:.4f},  gen_cls_acc: {:.4f}, real_cls_acc: {:.4f}, real_acc: {:.4f}, fake_acc: {:.4f}'\
                        .format(epoch+1, d_val_loss[0], d_val_loss[1], d_val_loss[2], d_val_loss[3], 
                                g_val_loss[0], g_val_loss[1], g_val_loss[2], g_val_loss[3], 
                                img_acc, gen_cls_acc, real_cls_acc, real_validate_acc, fake_validate_acc)
            print(log)
            
            # save model per epoch
            save_model(self.generator, 'face_generator_epoch{:02d}-acc{:.4f}-g_cls{:.4f}-r_cls{:.4f}'.format(epoch+1, img_acc, gen_cls_acc, real_cls_acc) )
            save_model(self.discriminator, 'face_discriminator_epoch{:02d}-acc{:.4f}-g_cls{:.4f}-r_cls{:.4f}'.format(epoch+1, img_acc, gen_cls_acc, real_cls_acc) )
            
            # test the model
            self.test_gan(epoch)
            
            # update learning rate
            if (epoch+1) % 1 == 0:
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
            ds = dataset.load_celeba('CelebA', batch_size, part=part, consumer = 'translator', smallbatch = 10)
            
            element = ds.make_one_shot_iterator().get_next()
            sess = K.get_session()
            imgs, labels = sess.run(element)
            
            labels = 1 - labels
            print(labels)
            rec_imgs = self.generator.predict([imgs, labels])
            src_real, _, cls_real = self.discriminator.predict(imgs)
            src_fake, _, cls_fake = self.discriminator.predict(rec_imgs)
            for r, f, sr, cr, sf, cf in zip(imgs, rec_imgs, src_real, cls_real, src_fake, cls_fake):
                plot_images([img_renorm(r), img_renorm(f)])
                print('real: ' + str(sr) + ' cls ' + str(cr) + '   fake: ' + str(sf) + ' cls ' + str(cf))
                


gan = FaceGAN(learning_rate = 0.0001, generator_norm_func = BatchNormalization, discriminator_norm_func = InstanceNormalization)
tf.logging.set_verbosity(tf.logging.ERROR)
gan.train(epochs=_epochs)
