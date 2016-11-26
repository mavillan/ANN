import numpy as np
# fixing the seed
np.random.seed(3)

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(8)

import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding

import sys
sys.path.append('/user/m/marvill/ANN/')
from keras_helper import load_keras_model as load
from keras_helper import save_keras_model as save

# directory where models will be saved
base_dir = '/user/m/marvill/ANN/tarea3/models/'



if __name__=='__main__':
    embedding_length = 32
    top_words = 5000

    # building the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=500))
    model.add(Flatten())
    model.add(Dense(4*embedding_length, activation='relu'))
    model.add(Dense(2*embedding_length, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fitting the model
    hist = model.fit(X_tr, y_tr, nb_epoch=50, verbose=0, validation_data=(X_val, y_val))
    # saving the model
    save(model, 'MLP_embedding{0}_tw{1}'.format(embedding_length, top_words), base_dir=base_dir)

