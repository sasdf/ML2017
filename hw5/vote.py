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
import pickle

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

class Sum(Layer):
    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

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

def f1(ans, pred, thresh):
    pred = (pred > thresh).sum(axis=0) > (pred.shape[0] // 2)
    tp = np.sum(ans*pred, axis=-1)
    fp = np.sum((1-ans)*pred, axis=-1)
    fn = np.sum(ans*(1-pred), axis=-1)
    #  print("tp: %d, fp: %d, fn: %d" %(tp.sum(),fp.sum(),fn.sum()))
    pr = tp / (tp + fp + 1e-20)
    re = tp / (tp + fn + 1e-20)
    return (2 * pr * re / (pr + re + 1e-20)).mean()


#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    #  (Y_data,X_data,tag_list) = read_data(train_path,True)
    #  print (tag_list)
    tag_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_test
    #  all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    if False:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_corpus)
    else:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    word_index = tokenizer.word_index
    
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    #  train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    #  train_sequences = pad_sequences(train_sequences)
    #  max_article_length = train_sequences.shape[1]
    max_article_length = 306
    #  print ("%d" % max_article_length)
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    #  train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    #  (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    num_words = len(word_index) + 1

    ### build model
    #  y_pred = []
    t_pred = []
    #  v_pred = []
    for path in os.listdir('.'):
        if path.endswith('.hdf5'):
            print ("Loading %s" % path)
            print ('Building model.')
            model = Sequential()
            e = Embedding(num_words,
                embedding_dim,
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
           
            model.load_weights(path)

            print ('Predicting.')
            #  v_pred.append(model.predict(X_val))
            #  y_pred.append(model.predict(X_train))
            t_pred.append(model.predict(test_sequences))
    w = np.array([0.399]*38)
    #  y_pred = np.array(y_pred)
    #  v_pred = np.array(v_pred)
    t_pred = np.array(t_pred)
    Y_pred = t_pred.transpose((1,0,2))
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        pred = (Y_pred > w).sum(axis=1) > (Y_pred.shape[1]//2)
        for index,labels in enumerate(pred):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
