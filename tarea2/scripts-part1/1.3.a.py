import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"\

import sys
sys.path.insert(0, '/user/m/marvill/ANN/')
from keras_helper import *

import pickle
import time
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model, load_model, Sequential
from keras.optimizers import SGD
from keras.objectives import binary_crossentropy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM


###############################
## LOADING DATA
###############################
def load_helper(nval=1000):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # 0-1 scaling
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    # reshaping
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    # training / validation split
    X_val = X_train[-nval:]
    y_val = y_train[-nval:]
    X_train = X_train[:-nval]
    y_train = y_train[:-nval]
    # correct format
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return X_train, X_val, X_test, y_train, y_val, y_test    

X_train, X_val, X_test, y_train, y_val, y_test = load_helper()


if __name__=='__main__':
    ## PARAMETERS
    n_hidden_layer1 = 1000
    activation_layer1 = 'sigmoid'; decoder_activation_1 = 'sigmoid'
    n_hidden_layer2 = 1000
    activation_layer2 = 'sigmoid'; decoder_activation_2 = 'sigmoid'
    ## MODEL DEFINITION AND TRAINING
    model = Sequential()
    model.add(Dense(n_hidden_layer1, activation=activation_layer1, input_shape=(784,)))
    model.add(Dense(n_hidden_layer2, activation=activation_layer2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, nb_epoch=50, batch_size=25, shuffle=True, validation_data=(X_val, y_val))
    ## STORING RESULTS
    pickle.dump( hist.history, open( "mlp_768x1000x1000x10.hist", "wb" ) )
    save_keras_model(model, 'mlp_768x1000x1000x10')
