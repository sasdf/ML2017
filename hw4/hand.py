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
import scipy.ndimage as nd
from PIL import Image

np.random.seed(42)

#  with h5py.File('/tmp2/sasdf/hw4model.h5', 'a') as f:
    #  if 'optimizer_weights' in f.keys():
        #  del f['optimizer_weights']
model = load_model('../58.h5')

images = [ Image.open('hand/hand.seq%d.png' % i) for i in range(100, 482) ]
#  data = [ 
            #  np.take(
                #  np.take(
                    #  np.array(im),
                    #  range(24, 480, 48),
                    #  axis=0
                #  ),
                #  range(25, 512, 51),
                #  axis=1
            #  ).flatten()
            #  for im in images
        #  ]
data = [ 
            nd.zoom(
                np.take(
                    np.array(im),
                    range(16, 496),
                    axis=1
                ),
                1.0/48
            ).flatten()
            for im in images
        ]
for im in images:
    im.close()

from sklearn.decomposition import PCA
testx = []
comp = 60
m = PCA(n_components=100)
t = m.fit_transform(data)[:,:comp]
t = np.repeat(t, comp, 0).reshape((t.shape[0], comp, comp), order='C')
weig = np.delete(np.tril(t), range(0,comp,2), 1)
pred = weig @ (m.components_[:comp])
diff = (pred.transpose((1,2,0)) - (data - m.mean_).T).transpose((2,0,1))
dist = (diff ** 2).sum(axis=2)**0.5
#  dist = np.abs(diff).mean(axis=2)
loss = np.abs(dist).mean(axis=0)

print("Loss: %f" %(loss.mean()))
l = m.explained_variance_
#  print (loss)
testx.append(np.concatenate([loss, l, m.explained_variance_ratio_]))
testx = np.array(testx)

opt = model.predict(testx, batch_size=42)
opt = np.clip(opt, np.log(1), np.log(60)).reshape(opt.shape[0])
opt = np.round(np.exp(opt))
print(opt[0])

#  from IPython import embed
#  embed()
