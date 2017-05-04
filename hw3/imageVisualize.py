memory_fraction = 0.05
import subprocess
import re
import time
import sys
print ('Checking Memory')
while True:
    m = subprocess.check_output(['nvidia-smi', '-q', '-d', 'MEMORY'])
    m = str(m).split('\n')
    free = [ re.search(r'Free\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    total = [ re.search(r'Total\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    free = [int(x.group(1)) for x in free if x][0]
    total = [int(x.group(1)) for x in total if x][0]
    need = total * (memory_fraction+0.05)
    sys.stdout.write('\33[2K\rTotal: %5d, Free: %5d, Need: %5d ' % (total, free, need))
    sys.stdout.flush()
    if free > need:
        print ('Enough Free memory available')
        break
    for i in range(3):
        time.sleep(.5)
        sys.stdout.write('.')
        sys.stdout.flush()


import os
import argparse
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
set_session(tf.Session(config=config))

import numpy as np
from keras.models import *
from keras.layers.core import *
from keras.layers import *
from keras.layers.noise import *
from keras.regularizers import l2
from keras.optimizers import *
from keras.layers.normalization import *
from keras.utils import np_utils
from keras.datasets import mnist
import sys, os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU
import gzip
from scipy import ndimage
import functools as fn
from keras.utils import plot_model

emotion_classifier = load_model('685.h5')
def loadData(p):
    f = open(p)
    ls = f.readlines()[1:]
    l2 = [ x.strip().split(',') for x in ls ]
    l3 = [ [i] + x.split(' ') for i, x in l2 ]
    return np.array(l3).astype('float32')
np.random.seed(42)

def preprocess(data):
    data_id = data[:, :1]
    data = data[:, 1:] / 255
    
    # print "Gamma encoding"
    # data = data ** 0.45

    #  print "Convert to CIE Lab colorspace"
    #  data = 116 * ( (data**(1.0/3))*(data > (6.0/29)**3) + (((29.0/6)**2)*data/3.0 + 16.0/116)*(data <= (6.0/29)**3) ) - 16
    #  data = data / 100
    
    print ("Adjust Contrast")
    data = np.clip(((data - 0.5) * 1.5 + 0.5), 0, 1)

    #  data = (data - 0.5) / 0.5
    #  data = data * (np.abs(data)**1.5) * 0.5 + 0.5
    
    # data = (data - 0.5) / 0.5
    # data = (((1.0 - np.abs(data)) ** 1.5) - 1.0) * np.sign(data)
    # data = data * 0.5 + 0.5

    # print "Gamma correction 1.5"
    # data = data ** 1.5

    #  print "Gamma correction 2.2"
    #  data = data ** 2.2
    
    # print "Convert from CIE Lab colorspace"
    # data = (data * 100 - 16) / 116
    # data = (data ** 3)*(data > (6.0/29)) + ((data - (16.0/116))*3*((6.0/29)**2))*(data <= (6.0/29))

    print ("Unbias")
    data = (data - 0.5) / 0.5

    #  print ("FFT")
    #  data = data.reshape(data.shape[0],48,48)
    #  daff = np.array(map(np.fft.fft2, data)).reshape(data.shape[0],48,48,1)
    #  dafi = daff.imag
    #  daff = daff.real
    #  data = data.reshape(data.shape[0],48,48,1)

    #  print "Blur"
    #  data = data.reshape(data.shape[0],48,48)
    #  blur = np.array(map(fn.partial(ndimage.gaussian_filter, sigma=0.8), data))
    #  data = data * 0 + 1 * blur
    #  data = np.clip(data, -1, 1)
    #  data = data.reshape(data.shape[0],48,48,1)

    print ("Sharpen")
    data = data.reshape(data.shape[0],48,48)
    blur = np.array(list(map(fn.partial(ndimage.gaussian_filter, sigma=0.8), data)))
    data = data + 1 * (data - blur)
    data = np.clip(data, -1, 1)
    data = data.reshape(data.shape[0],48,48,1)

    #  print "Transpose Image(?)"
    #  data = np.transpose(data, [0,2,1,3])
    return data_id, data, 0, 0

train = loadData(sys.argv[1])
np.random.shuffle(train)
train = train[7:8]
train = preprocess(train)

print ("PreprocessDone")

train_y, train_x, traff_x, trafi_x = train
train_y = np_utils.to_categorical(train_y, 7)



def adamEmpty(g):
    return {
                'm': np.zeros_like(g),
                'v': np.zeros_like(g),
                'b1t': 1,
                'b2t': 1,
           }


def adam(g, state):
    m = state['m'].copy()
    v = state['v'].copy()
    b1t = state['b1t']
    b2t = state['b2t']
    a = 0.001
    b1 = 0.9
    b2 = 0.999
    e = 10e-8
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*(g**2)
    b1t *= b1
    b2t *= b2
    c = a * (m/(1-b1t)) / ((v/(1-b2t))**0.5 + e)
    return (c, {
                'm': m,
                'v': v,
                'b1t': b1t,
                'b2t': b2t,
           })


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    loss, grad = iter_func([input_image_data,False])
    cache = adamEmpty(grad)
    best_img = input_image_data
    best_los = loss
    for i in range(num_step):
        f = (num_step+1)**0.2
        loss, grad = iter_func([input_image_data,False])
        if (abs(loss) < abs(best_los)):
            best_img = input_image_data
            best_los = loss
        step, cache = adam(grad, cache)
        input_image_data += step  #* (1.0/f)
        input_image_data  = np.clip(input_image_data, -1.0, 1.0)
        if (i > 100 and abs(loss) < 0.03):
            break
    return (best_img, best_los)


layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
input_img = emotion_classifier.input
#  from pprint import pprint
#  emotion_classifier.summary()
#  exit(0)
name_ls = ["conv2d_4"]
collect_layers = [ layer_dict[name].output for name in name_ls ]
#  np.random.seed(43)
for cnt, c in enumerate(collect_layers):
    filter_imgs = []
    for filter_idx in range(64):
        input_img_data = train_x
        target = K.mean(c[:, :, :, filter_idx])
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function([input_img, K.learning_phase()], [target, grads])

        ###
        "You need to implement it."
        print filter_idx
        filter_imgs.append(grad_ascent(800, input_img_data, iterate))
        ###

    fig = plt.figure(figsize=(14, 8))
    for i in range(64):
        ax = fig.add_subplot(64//16, 16, i+1)
        ax.imshow(filter_imgs[i][0].reshape(48,48), cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
        plt.tight_layout()
    fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], 800))
    #  fig.savefig('imageVis.png')
    plt.show()
