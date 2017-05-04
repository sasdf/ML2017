memory_fraction = 0.1
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
#  train = train[4:5]
train = preprocess(train)

print ("PreprocessDone")

train_y, train_x, traff_x, trafi_x = train
train_y = np_utils.to_categorical(train_y, 7)

opt = emotion_classifier.predict(train_x, batch_size=42)
pre = np.array([np.where(a == a.max())[0][0] for i, a in enumerate(opt)])
ans = np.array([np.where(a == a.max())[0][0] for i, a in enumerate(train_y)])
correct = opt[pre == ans]
m = np.array([a.max() for a in correct])
train_x = train_x[pre == ans]
imgs = train_x[m == 1]

fig = plt.figure(figsize=(14, 8))
for idx, x in enumerate(imgs):
    img = x.reshape(48,48)
    ax = fig.add_subplot(15, 10, idx+1)
    ax.imshow(img, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(15, 10))
for idx, x in enumerate(imgs):
    train_x = x.reshape(1,48,48,1)
    input_img = emotion_classifier.input
    img_ids = ["image ids from which you want to make heatmaps"]

    val_proba = emotion_classifier.predict(train_x)
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classifier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    heatmap = None
    '''
    Implement your heatmap processing here!
    hint: Do some normalization or smoothening on grads
    '''
    img = train_x
    heatmap = fn([img, False])[0]
    #  for i in range(100):
        #  img += fn([img, False])[0] * 0.1
        #  heatmap += fn([img, False])[0]
        #  heatmap = np.clip(heatmap, -1, 1)
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
    heatmap = abs(np.clip(heatmap, -1, 1))

    heatmap = heatmap.reshape(48,48)
    thres = 0.5
    see = train_x.reshape(48, 48)
    #  plt.figure()
    see[np.where(heatmap <= thres)] = np.mean(see)

    ax = fig.add_subplot(15, 10, idx+1)
    ax.imshow(see, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.show()
