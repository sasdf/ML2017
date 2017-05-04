#!/usr/bin/env python
# -- coding: utf-8 --

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

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
train = train[:len(train)//10]
train = preprocess(train)

print ("PreprocessDone")

train_y, train_x, traff_x, trafi_x = train

model = load_model('685.h5')

opt = model.predict(train_x, batch_size=42)
predict = np.array(list(map(lambda a: np.where(a == a.max())[0][0], opt)))
answer = train_y
pair = zip(predict, answer)
pair = ( (x, a) for x, a in pair if a != x )
predict, answer = zip(*pair)

np.set_printoptions(precision=2)
conf_mat = confusion_matrix(answer,predict)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
