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

def gaussian_noise(X_train, X_val, X_test, devst=0.25):
    noise_mask = devst*np.random.standard_normal(size=X_train.shape)
    X_train_noisy = X_train + noise_mask
    noise_mask = devst*np.random.standard_normal(size=X_val.shape)
    X_val_noisy = X_val + noise_mask
    noise_mask = devst*np.random.standard_normal(size=X_test.shape)
    X_test_noisy = X_test + noise_mask
    return X_train_noisy, X_val_noisy, X_test_noisy 


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
    activation_layer1 = 'relu'; decoder_activation_1 = 'sigmoid'
    n_hidden_layer2 = 1000
    activation_layer2 = 'relu'; decoder_activation_2 = 'sigmoid'
    loss_ = 'binary_crossentropy'
    optimizer_ = SGD(lr=1.0)
    epochs_ = 50
    batch_size_ = 25
    
    # Data with gaussian noise
    X_train_noisy, X_val_noisy, X_test_noisy = gaussian_noise(X_train, X_val, X_test)

    ### DENOISING AUTOENCODER 1
    input_img1 = Input(shape=(784,))
    encoded1 = Dense(n_hidden_layer1, activation=activation_layer1)(input_img1)
    decoded1 = Dense(784, activation=decoder_activation_1)(encoded1)
    autoencoder1 = Model(input=input_img1, output=decoded1)
    encoder1 = Model(input=input_img1, output=encoded1)
    autoencoder1.compile(optimizer=optimizer_, loss=loss_)
    autoencoder1.fit(X_train, X_train_noisy, nb_epoch=epochs_, batch_size=batch_size_, 
                     shuffle=True, validation_data=(X_val, X_val_noisy))

    ### DENOISING AUTOENCODER 2
    # FORWARD PASS DATA THROUGH FIRST ENCODER
    X_train_encoded1 = encoder1.predict(X_train) 
    X_val_encoded1 = encoder1.predict(X_val)
    X_test_encoded1 = encoder1.predict(X_test)
    # adding gaussian noise also to encoded data
    X_train_noisy_encoded1, X_val_noisy_encoded1, X_test_noisy_encoded1 = gaussian_noise(X_train_encoded1, X_val_encoded1,
                                                                                         X_test_encoded1)
    input_img2 = Input(shape=(n_hidden_layer1,))
    encoded2 = Dense(n_hidden_layer2, activation=activation_layer2)(input_img2)
    decoded2 = Dense(n_hidden_layer2, activation=decoder_activation_2)(encoded2)
    autoencoder2 = Model(input=input_img2, output=decoded2)
    encoder2 = Model(input=input_img2, output=encoded2)
    autoencoder2.compile(optimizer=optimizer_, loss=loss_)
    autoencoder2.fit(X_train_encoded1, X_train_noisy_encoded1, nb_epoch=epochs_, batch_size=batch_size_,
                     shuffle=True, validation_data=(X_val_encoded1, X_val_noisy_encoded1))
    encoded_input2 = Input(shape=(n_hidden_layer2,))

    ### FINE TUNNING
    model = Sequential()
    model.add( Dense(n_hidden_layer1, activation=activation_layer1, input_shape=(784,)) )
    model.layers[-1].set_weights( autoencoder1.layers[1].get_weights() )
    model.add( Dense(n_hidden_layer2, activation=activation_layer2) )
    model.layers[-1].set_weights( autoencoder2.layers[1].get_weights() )
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer_, loss='binary_crossentropy', metrics=['accuracy'])
    # saving net before fine tunning
    save_keras_model(model, 'mlp_768x1000x1000x10_pretrain_dae_relu')
    model.fit(X_train, y_train, nb_epoch=20, batch_size=25, shuffle=True, validation_data=(X_val, y_val))
    # saving net after fine tunning
    save_keras_model(model, 'mlp_768x1000x1000x10_finetunning_dae_relu')
