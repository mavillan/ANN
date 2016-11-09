import os
import sys
import numpy as np
import cPickle
import pickle

from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model, load_model, save_model, Sequential
from keras.optimizers import SGD
from keras.objectives import binary_crossentropy
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def load_NORB_train_val(PATH, batch=1):
    f = os.path.join(PATH, 'data_batch_{0}'.format(batch))
    datadict = unpickle(f)
    X = datadict['data'].T
    Y = np.expand_dims(np.array(datadict['labels']), axis=1)
    Z = np.concatenate((X,Y),axis=1)
    np.random.shuffle(Z)
    return Z[5832:,0:-1], Z[5832:,-1],Z[:5832,0:-1], Z[:5832,-1]

def load_NORB_test(PATH):
    xt = []
    yt = []
    for b in range(11,13):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        datadict = unpickle(f)
        X = datadict['data'].T
        Y = np.expand_dims(np.array(datadict['labels']), axis=1)
        Z = np.concatenate((X,Y),axis=1)
        np.random.shuffle(Z)
        xt.append(Z[:,0:-1])
        yt.append(Z[:,-1])
    Xt   = np.concatenate(xt)
    Yt   = np.concatenate(yt)
    del xt,yt
    return Xt, Yt

def data_transform(X, normalize=True, a=None, b=None):
    if normalize: return StandardScaler().fit_transform(X)
    else: return X*(b-a) + a 

def build_model(activation='relu'):
    model = Sequential()
    model.add(Dense(4000, input_dim=2048, activation=activation))
    model.add(Dense(2000, activation=activation))
    model.add(Dense(6, activation='softmax'))
    sgd = SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


### LOADING TEST DATA
X_test, y_test = load_NORB_test('/user/m/marvill/ANN/tarea2/data_part2/')
X_test_scaled = data_transform(X_test)
Y_test = np_utils.to_categorical(y_test, 6)


if __name__=='__main__':
    activation = sys.argv[1]
    model = build_model(activation)

    acc_list = []

    for i, theta in enumerate(np.linspace(0.1, 1., 10)):
        X_train, y_train, X_val, y_val = load_NORB_train_val('/user/m/marvill/ANN/tarea2/data_part2/', i+1)
        # scalling data
        X_train_scaled = data_transform(X_train)
        X_val_scaled = data_transform(X_val)
        # categorizing labels
        Y_train = np_utils.to_categorical(y_train, 6)
        Y_val = np_utils.to_categorical(y_val, 6)
        # model training
        model.fit(X_train_scaled, Y_train, batch_size=100, validation_data=(X_val_scaled, Y_val), nb_epoch=10)
        acc = model.evaluate(X_test_scaled, Y_test, verbose=0)
        acc_list.append(acc)
    pickle.dump(acc_list, open('mlp_relu_acc','wb'))
