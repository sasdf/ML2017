#  memory_fraction = 0.2
#  import subprocess
#  import re
#  import time
#  import sys
#  print ('Checking Memory')
#  while True:
    #  m = subprocess.check_output(['nvidia-smi', '-q', '-d', 'MEMORY'])
    #  m = str(m).split('\n')
    #  free = [ re.search(r'Free\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    #  total = [ re.search(r'Total\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    #  free = [int(x.group(1)) for x in free if x][0]
    #  total = [int(x.group(1)) for x in total if x][0]
    #  need = total * (memory_fraction+0.05)
    #  sys.stdout.write('\33[2K\rTotal: %5d, Free: %5d, Need: %5d ' % (total, free, need))
    #  sys.stdout.flush()
    #  if free > need:
        #  print ('Enough Free memory available')
        #  break
    #  for i in range(3):
        #  time.sleep(.5)
        #  sys.stdout.write('.')
        #  sys.stdout.flush()




#  import tensorflow as tf
#  from keras.backend.tensorflow_backend import set_session
#  config = tf.ConfigProto()
#  config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
#  set_session(tf.Session(config=config))


import sys, csv, string
import random
import numpy as np
import pandas as pd
import keras.backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.regularizers import *
from keras.losses import *
from keras.engine.topology import Layer
from keras.initializers import *
from keras.activations import *
np.random.seed(242)
random.seed(242)

#  train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[1]+'test.csv')

#  train = train.sample(frac=1)

#  trainu = train['UserID'].values
#  trainm = train['MovieID'].values
#  trainy = train['Rating'].values

testu = test['UserID'].values
testm = test['MovieID'].values

xu = Input((1,))
xm = Input((1,))

yu = xu
ym = xm

yu = Embedding(
    6040,
    1000,
    input_length=1,
    trainable=True,
    #  mask_zero=True,
    embeddings_initializer = 'glorot_normal',
    #  embeddings_initializer=TruncatedNormal(2, 0.9)
    #  embeddings_regularizer=l2(1e-7),
)(yu)

yu = Flatten()(yu)

ym = Embedding(
    3952,
    1000,
    input_length=1,
    trainable=True,
    #  mask_zero=True,
    embeddings_initializer = 'glorot_normal',
    #  embeddings_initializer=TruncatedNormal(2, 0.9)
    embeddings_regularizer=l2(1e-7),
)(ym)

ym = Flatten()(ym)

y = dot([yu, ym], 1)


model = Model(inputs = (xu,xm), outputs = y)
model.load_weights('model.h5')
testy = model.predict([testu, testm], batch_size=3000)
opt = pd.read_csv('SampleSubmisson.csv')
opt['Rating'] = testy
opt.to_csv(sys.argv[2], index=False)
