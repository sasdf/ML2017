import numpy as np

from sklearn.decomposition import PCA
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
from scipy import ndimage
from keras.utils import plot_model

opt = []
datas = np.load(sys.argv[1])
np.random.seed(42)
for k in datas.keys():
    print("index: %s" % k)
    # if we want to generate data with intrinsic dimension of 10
    data = datas[k]
    #  np.random.shuffle(data)
    data = data[:10000]


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

    print("Index: %s, Loss: %f" %(k, loss.mean()))
    l = m.explained_variance_
    #  print (loss)
    opt.append(np.concatenate([[int(k)], loss, l, m.explained_variance_ratio_]))

test = np.array(opt)

np.random.seed(42)

#  with h5py.File('model.h5', 'a') as f:
    #  if 'optimizer_weights' in f.keys():
        #  del f['optimizer_weights']
model = load_model('model.h5')

testid = test[:, 0]
testx = test[:,1:]

opt = model.predict(testx, batch_size=42)
opt = np.clip(opt, np.log(1), np.log(60)).reshape(opt.shape[0])
opt = np.log(np.round(np.exp(opt)))
opt = zip(testid, opt)
res = 'SetId,LogDim\n'
res += ''.join([ '%d, %f\n'%(i, a) for i, a in opt ])
f = open(sys.argv[2], 'w')
f.write(res)
f.close()
