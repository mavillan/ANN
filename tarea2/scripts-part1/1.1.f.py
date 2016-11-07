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
from keras.models import Model, load_model
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


###################################
## AUTOENCODER TRAINING FUNCTION
###################################
def train_deep_ae(X_train, X_val, layer_sizes=[1000, 500, 250, 2], activation='relu'):
    layers_list = []
    layers_list.append( Input(shape=(784,)) )
    # encoding model
    for size in layer_sizes:
        layers_list.append( Dense(size, activation=activation)(layers_list[-1]) )
    encoded = layers_list[-1]
    # decoding model
    for size in layer_sizes[-2::-1]:
        layers_list.append( Dense(size, activation=activation)(layers_list[-1]) )
    layers_list.append( Dense(784, activation='sigmoid')(layers_list[-1]) )
    # encoder and autoencoder model
    encoder = Model(input=layers_list[0], output=encoded)
    autoencoder = Model(input=layers_list[0], output=layers_list[-1])
    # training step
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train, nb_epoch=50, batch_size=25, shuffle=True, validation_data=(X_val, X_val))
    # saving results
    save_keras_model(encoder, 'enc_{0}L_{1}d'.format(len(layer_sizes), layer_sizes[-1]))
    save_keras_model(autoencoder, 'ae_{0}L_{1}d'.format(len(layer_sizes), layer_sizes[-1]))
    return autoencoder

if __name__=='__main__':
    layer_sizes = list(map(int, sys.argv[1].strip().split('-')))
    train_deep_ae(X_train, X_val, layer_sizes=layer_sizes)

