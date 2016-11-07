import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
import numpy as np
import random
import time
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

def save_keras_model(model, model_name):
    model_name = model_name.split('.')[0]
    # serialize model to JSON
    model_json = model.to_json()
    with open('{0}.json'.format(model_name), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('{0}.h5'.format(model_name))

if __name__=='__main__':
    path = '/user/a/asalinas/ANN/'
    name = 'outputs/5-2'
    train_data = sio.loadmat(path+'train_32x32.mat')
    test_data = sio.loadmat(path+'test_32x32.mat')
    X_train = train_data['X'].T
    y_train = train_data['y'] - 1
    X_test = test_data['X'].T
    y_test = test_data['y'] - 1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    n_classes = len(np.unique(y_train))
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    n_channels = X_train.shape[1]
    n_rows = X_train.shape[2]
    n_cols = X_train.shape[3]
    cf = 5
    cp = 2
    model = Sequential()
    model.add(Convolution2D(16, cf, cf, border_mode='same', activation='relu',input_shape=(n_channels, n_rows, n_cols)))
    model.add(MaxPooling2D(pool_size=(cp, cp)))
    model.add(Convolution2D(512, cf, cf, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(cp, cp)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    start_time = time.time()
    hist = model.fit(X_train, Y_train, batch_size=1280, nb_epoch=12, verbose=0, validation_data=(X_test, Y_test))
    elapsed_time = time.time() - start_time
    save_keras_model(model, path+name)
    arch = open(path+name,'w')
    arch.write('Elapsed time:'+str(elapsed_time))
    arch.write('\nHistory:'+str(hist.history.items()))
    arch.close()
