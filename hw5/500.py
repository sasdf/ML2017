import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random

np.random.seed(42)
random.seed(42)

import json
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 300
nb_epoch = 1000
batch_size = 64


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    with open('/tmp2/sasdf/glove.840B.300d.pkl', 'rb') as f:
        import pickle
        return pickle.load(f)
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

import keras.backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.regularizers import *
from keras.engine.topology import Layer
from keras.initializers import *
from keras.activations import *

class Exp(Layer):
    def call(self, x):
        return K.exp(x)

class Sum(Layer):
    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class Power(Layer):
    def __init__(self, power, **kwargs):
        self.power = power
        super(Power, self).__init__(**kwargs)

    def call(self, x):
        return K.pow(x,self.power)

class EmbeddingMask(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(EmbeddingMask, self).__init__(**kwargs)

    def build(self, input_shape):
        self.repeat_dim = input_shape[2]
        super(EmbeddingMask, self).build(input_shape)

    def compute_mask(self, input_shape, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return mask

class Masked(Layer):
    def __init__(self, preserve=False, **kwargs):
        self.support_mask = True
        self.preserve = preserve
        super(Masked, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x * mask

    def compute_mask(self, input_shape, mask=None):
        if self.preserve:
            return mask
        return None

class MaskedMean(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedMean, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[2]
        super(MaskedMean, self).build(input_shape)

    def call(self, x, mask=None):
        n = K.sum(mask, axis=1, keepdims=False)
        x = x * mask
        mean = K.sum(x, axis=1, keepdims=False) / n
        return mean

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

class MaskedGMean(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedGMean, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[2]
        super(MaskedGMean, self).build(input_shape)

    def call(self, x, mask=None):
        n = K.sum(mask, axis=1, keepdims=False)
        x = x * mask
        x = K.relu(x) + K.epsilon()
        x = K.log(x)
        mean = K.sum(x, axis=1, keepdims=False) / n
        ret = K.exp(mean)
        return ret

    def compute_mask(self, input_shape, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
def f1loss(true, pred):
    thresh = 0.4
    pred -= thresh
    norm = K.cast(pred > 0, dtype='float32') * (1.0-thresh) + K.cast(pred <= 0, dtype='float32') * thresh
    pred /= norm
    pred = K.pow(K.abs(pred), 1) * pred
    pred *= 0.5
    pred += 0.5
    tp = K.sum(true * pred, axis=-1)
    fp = K.sum((1.0-true) * pred, axis=-1)
    fn = K.sum(true * (1.0-pred), axis=-1)
    pr = tp / (tp + fp + K.epsilon()) + K.epsilon()
    re = tp / (tp + fn + K.epsilon()) + K.epsilon()
    return 1 / K.mean(2 * pr * re / (pr + re)) ** 5

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('/tmp2/sasdf/glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    np.random.seed(42)
    random.seed(42)
    ### build model
    y_pred = []
    for i in range(50):
        print ("Round: %d" % i)
        print ('Building model.')
        model = Sequential()
        e = Embedding(num_words,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_article_length,
            mask_zero = True,
            trainable=False,
            embeddings_regularizer=l2(1e-6)
        )
        model.add(e)
        model.add(EmbeddingMask())
        model.add(Masked())
        model.add(Sum())
        #  model.add(GRU(128,activation='tanh',dropout=0.25))
        model.add(Dense(256,activation='elu', kernel_regularizer=l2(1e-6)))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='elu', kernel_regularizer=l2(1e-6)))
        model.add(Dropout(0.5))
        model.add(Dense(38,activation='sigmoid'))
        model.summary()

        adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
        model.compile(
            #  loss=f1loss,
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=[f1_score]
        )
       
        earlystopping = EarlyStopping(monitor='val_f1_score', patience = 15, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath='round%d.hdf5'%i,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_f1_score',
                                     mode='max')
        
        #  model.load_weights('bestpre.hdf5')
        try:
            hist = model.fit(X_train, Y_train, 
                             validation_data=(X_val, Y_val),
                             epochs=nb_epoch, 
                             batch_size=batch_size*2,
                             callbacks=[earlystopping,checkpoint])
        except KeyboardInterrupt:
            cont = ''
            print("")
            while (cont != 'y' and cont != 'n'):
                cont = input("Keep Running?[Y/n]")
                cont = cont.lower().strip()
            if cont == 'n':
                break
            pass
        
        model.load_weights('round%d.hdf5'%i)

        y_pred.append(model.predict(test_sequences))
        #  thresh = 0.4
        #  with open(output_path,'w') as output:
            #  print ('\"id\",\"tags\"',file=output)
            #  Y_pred_thresh = (Y_pred > thresh).astype('int')
            #  for index,labels in enumerate(Y_pred_thresh):
                #  labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
                #  labels_original = ' '.join(labels)
                #  print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    from IPython import embed
    embed()

if __name__=='__main__':
    main()
