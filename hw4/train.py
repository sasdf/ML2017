memory_fraction = 0.3
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
import pickle

np.random.seed(41)

with open(sys.argv[1], 'rb') as f:
    data = np.array(pickle.load(f))
    np.random.shuffle(data)
    sp = min(len(data)//5, 200)
    vali = data[:sp]
    valiy = np.log(vali[:, 0] + 1)
    valix = vali[:,1:]
    data = data[sp:]
    datay = data[:, 0]
    datax = data[:,1:]

    datay = np.log(datay + 1)

y = x = Input(shape = (230,))
size = 16 
for i in range(size):
    y = Dense(128*int((size-i)**0.5),
        #  padding = 'same',
        kernel_initializer = 'glorot_normal',
        #  kernel_regularizer=l2(0.000000000001),
    )(y)
    y = LeakyReLU()(y)
    #  y = BatchNormalization()(y)
    #  y = Dropout(0.01)(y)

y = Dense(1,
    kernel_initializer = 'glorot_normal',
    #  kernel_regularizer=l2(0.000001),
    #  activation = 'softmax',
)(y)

model = Model(inputs = x, outputs = y)
model.summary()
#  plot_model(model, to_file='model.png')

def roundMAE(true, pred):
    pred = K.clip(pred, np.log(1), np.log(60))
    pred = K.log(K.round(K.exp(pred)))
    return K.mean(K.abs(pred - true), axis=-1)

model.compile(loss='mean_absolute_error',optimizer='adam', metrics=[roundMAE])

#  model = load_model('modelpre.h5')
for i in range(1,14):
    model.fit(
            datax, datay,
            batch_size = 5 * int((i+2) ** 3),
            epochs=i,
            validation_data=(valix, valiy),
            callbacks=[
                ModelCheckpoint('model.h5', monitor='val_roundMAE', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            ])
model.fit(
        datax, datay,
        batch_size = 13000,
        epochs=100,
        validation_data=(valix, valiy),
        callbacks=[
            ModelCheckpoint('model.h5', monitor='val_roundMAE', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ])

with h5py.File('model.h5', 'a') as f:
    if 'optimizer_weights' in f.keys():
        del f['optimizer_weights']
model = load_model('model.h5', custom_objects={'roundMAE': roundMAE})

#  with open(sys.argv[2], 'rb') as f:
    #  test = np.array(pickle.load(f))
    #  testid = test[:, 0]
    #  testx = test[:,1:]

#  opt = model.predict(testx, batch_size=42)
#  opt = np.clip(opt, np.log(1), np.log(60)).reshape(opt.shape[0])
#  opt = zip(testid, opt)
#  res = 'SetId,LogDim\n'
#  res += ''.join([ '%d, %f\n'%(i, a) for i, a in opt ])
#  f = open(sys.argv[3], 'w')
#  f.write(res)
#  f.close()

#  from IPython import embed
#  embed()
