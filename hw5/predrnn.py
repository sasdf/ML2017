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
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


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
    #  (Y_data,X_data,tag_list) = read_data(train_path,True)
    tag_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']
    (_, X_test,_) = read_data(test_path,False)
    #  all_corpus = X_data + X_test
    all_corpus = X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    if False:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_corpus)
        with open("tokenizerRNN.pkl", 'wb') as f:
            pickle.dump(tokenizer, f)
    else:
        with open("tokenizerRNN.pkl", 'rb') as f:
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
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    #  train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    #  (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get mebedding matrix from glove
    #  print ('Get embedding dict from glove.')
    #  embedding_dict = get_embedding_dict('/tmp2/sasdf/glove.6B.%dd.txt'%embedding_dim)
    #  print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    #  print ('Create embedding matrix.')
    #  embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        #  weights=[embedding_matrix],
                        input_length=max_article_length,
                        mask_zero = True,
                        trainable=False))
    model.add(GRU(128,activation='tanh',dropout=0.25))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(
        #  loss=f1loss,
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=[f1_score]
    )
   
    #  earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
    #  checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 #  verbose=1,
                                 #  save_best_only=True,
                                 #  save_weights_only=True,
                                 #  monitor='val_f1_score',
                                 #  mode='max')
    
    #  hist = model.fit(X_train, Y_train, 
                     #  validation_data=(X_val, Y_val),
                     #  epochs=nb_epoch, 
                     #  batch_size=batch_size,
                     #  callbacks=[earlystopping,checkpoint])
    
    model.load_weights('rnn/best.hdf5')

    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
