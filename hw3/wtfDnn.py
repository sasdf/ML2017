memory_fraction = 0.10
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

    print ("FFT")
    data = data.reshape(data.shape[0],48,48)
    daff = np.array(map(np.fft.fft2, data)).reshape(data.shape[0],48,48,1)
    dafi = daff.imag
    daff = daff.real
    data = data.reshape(data.shape[0],48,48,1)

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
    return data_id, data, daff, dafi

test = loadData(sys.argv[2])
test = preprocess(test)

train = loadData(sys.argv[1])
np.random.shuffle(train)
valid = train[:len(train)//10]
train = train[len(train)//10:]
train = preprocess(train)
valid = preprocess(valid)

print ("PreprocessDone")

train_y, train_x, traff_x, trafi_x = train
valid_y, valid_x, valff_x, valfi_x = valid
test_id, test_x, teff_x, tefi_x = test
train_y = np_utils.to_categorical(train_y, 7)
valid_y = np_utils.to_categorical(valid_y, 7)


yo = xo = Input(shape = (48, 48, 1))
yo = Flatten()(yo)
#  yo = GaussianNoise(0.1)(yo)
for i in range(5):
    yo = Dense(256*(5-i),
        #  padding = 'same',
        kernel_initializer = 'glorot_normal',
        #  kernel_regularizer=l2(0.00000001),
    )(yo)
    yo = LeakyReLU()(yo)
    yo = BatchNormalization()(yo)
    yo = Dropout(0.1)(yo)

y = yo
y = Dense(7,
    kernel_initializer = 'glorot_normal',
    #  kernel_regularizer=l2(0.000001),
    activation = 'softmax',
)(y)

model = Model(inputs = xo, outputs = [y])
model.summary()
#  plot_model(model, to_file='model.png')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#  model = load_model('modelpre.h5')

train_gen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

train_gen.fit(train_x)

history = History()

model.fit_generator(
        #  train_x, train_y,
        train_gen.flow(train_x, train_y, batch_size=100),
        steps_per_epoch=len(train_x)//100,
        #  batch_size = 30,
        epochs=300,
        validation_data=(valid_x, valid_y),
        callbacks=[
            ModelCheckpoint('modelDNN.h5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
            history
        ])

hisfile = open("historyDNN", "w")
hist = {
    'trl': history.tr_losses,
    'vall': history.val_losses,
    'tra': history.tr_accs,
    'vala': history.val_accs
}
pickle.dump(hist, hisfile)
hisfile.close()

model = load_model('modelDNN.h5')

opt = model.predict(test_x, batch_size=42)
res = 'id,label\n'
for i, a in enumerate(opt):
    aa = np.where(a == a.max())[0][0]
    res += '{},{}\n'.format(i, aa)

f = open(sys.argv[3], 'w')
f.write(res)
f.close()

#  from IPython import embed
#  embed()
