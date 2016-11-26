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


def generate_model(top_words, embedding_length, n_lstm_units=100):
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, input_length=500))
    model.add(LSTM(n_lstm_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__=='__main__':
    top_words = int(sys.argv[1])

    # loading data
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words, seed=15)
    X_train = sequence.pad_sequences(X_train, maxlen=500)
    X_test = sequence.pad_sequences(X_test, maxlen=500)

    embedding_lengths = [8,16,32,64,128,256,512,1024]

    for length in embedding_lengths:
        model = generate_model(top_words=top_words , embedding_length=length, n_lstm_units=100)
        # fitting the model
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
        # saving the model
        save(model, 'lstm100_embbeding{0}_tw{1}'.format(length, top_words), base_dir=base_dir)
