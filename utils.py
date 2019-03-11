# -*- coding: utf-8 -*-
from __future__ import print_function 
import numpy as np 
import tensorflow as tf 
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def img_renorm(img):
    return (img + 1.0) / 2.0

def plot_image(input_images, rec_images, epoch = 0, save_image=True):
    for x, r in zip(input_images, rec_images):
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(r)
        plt.axis('off')
        if save_image:
            plt.savefig('image_pair'+ str(epoch) + '_' + str(time.time()) + '.jpg')
        plt.show()
        
def save_model(model, name):
    json_string = model.to_json()
    file = open(name + '.json', 'w') 
    file.write(json_string) 
    file.close() 
    model.save_weights(name + '.h5')

def load_model(name):
    from tensorflow.keras.models import model_from_json
    model = model_from_json(open(name + '.json', 'r').read())
    model.load_weights(name + '.h5')
    return model
    
#generate random index
def generate_rand_index():
    index=np.arange(10000)  
    np.random.shuffle(index)  
    print(index[0:20])
    
    np.save("validation_index.npy",index[0:5000])
    np.save("test_index.npy",index[5000:10000])
    
def load_index():
    index_v = np.load("validation_index.npy")
    index_t = np.load("test_index.npy")
    print(index_v[0:20])
    print(index_t[0:20])


def plot_images(images, save_image=True, show_image=True, filename = None):
    num = len(images)
    fig = plt.figure(figsize = (num*2.5,1*2.5))
    i = 1
    for image in images:
        plt.subplot(1, num, i)
        plt.imshow(image, aspect='auto')
        plt.axis('off')
        i += 1
    if save_image:
        if filename:
            plt.savefig(filename + '.jpg')
        else:
            plt.savefig('images'+ str(time.time()) + '.jpg')
    if show_image:
        plt.show()

def plot_image_list(imagelist, save_image=True):
    col = len(imagelist)
    row = len(imagelist[0])
    print(col)
    print(row)
    for i in range(row):
        fig = plt.figure(figsize = (3*2.5,1*2.5))
        for j in range(col):
            plt.subplot(1, col, j+1)
            plt.imshow(imagelist[j][i])
            plt.axis('off')
        if save_image:
            plt.savefig('image_list'+ str(time.time()) + '.jpg')
        plt.show()


def read_image(path):
    file = tf.read_file(path)
    image = tf.image.decode_and_crop_jpeg(file, [29, 9, 160, 160])
    image.set_shape((160, 160, 3))
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1
    image = image.eval(session=tf.Session())
    return image


def show_image(image):
    plt.imshow(image, interpolation='spline16')
    plt.show()
