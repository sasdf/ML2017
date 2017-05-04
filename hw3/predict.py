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

test = loadData(sys.argv[2])
test = preprocess(test)

#  train = loadData(sys.argv[1])
#  np.random.shuffle(train)
#  valid = train[:len(train)//10]
#  train = train[len(train)//10:]
#  train = preprocess(train)
#  valid = preprocess(valid)

print ("PreprocessDone")

#  train_y, train_x, traff_x, trafi_x = train
#  valid_y, valid_x, valff_x, valfi_x = valid
test_id, test_x, teff_x, tefi_x = test
#  train_y = np_utils.to_categorical(train_y, 7)
#  valid_y = np_utils.to_categorical(valid_y, 7)


model = load_model('685.h5')

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
