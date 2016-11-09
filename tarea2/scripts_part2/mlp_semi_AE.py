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
    model.add(Dense(40, input_dim=2048, activation=activation))
    model.add(Dense(20, activation=activation))
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
    n_batches = int(sys.argv[2])
    acc_list = []

    xtr_s = []
    ytr_s = []
    xval_s = []
    yval_s = []
    xtr_ns = []
    ytr_ns = []
    xval_ns = []
    yval_ns = []

    # loading data, and splitting it for supervised and no supervised training purposes
    for i in range(1, 11):
        if i<=n_batches:
            X_train, y_train, X_val, y_val = load_NORB_train_val('/user/m/marvill/ANN/tarea2/data_part2/', i)
            xtr_s.append(data_transform(X_train)); ytr_s.append(y_train)
            xval_s.append(data_transform(X_val)); yval_s.append(y_val)
        else:
            X_train, y_train, X_val, y_val = load_NORB_train_val('/user/m/marvill/ANN/tarea2/data_part2/', i)
            xtr_ns.append(data_transform(X_train)); ytr_ns.append(y_train)
            xval_ns.append(data_transform(X_val)); yval_ns.append(y_val)

    X_train_s = np.concatenate(xtr_s); y_train_s = np.concatenate(ytr_s)
    X_val_s = np.concatenate(xval_s); y_val_s = np.concatenate(yval_s)
    X_train_ns = np.concatenate(xtr_ns); y_train_ns = np.concatenate(ytr_ns)
    X_val_ns = np.concatenate(xval_ns); y_val_ns = np.concatenate(yval_ns)
    del xtr_s, ytr_s, xval_s, yval_s, xtr_ns, ytr_ns, xval_ns, yval_ns

    # categorizing targets
    Y_train_s = np_utils.to_categorical(y_train_s, 6)
    Y_train_ns = np_utils.to_categorical(y_train_ns, 6)
    Y_val_s = np_utils.to_categorical(y_val_s, 6)
    Y_val_ns = np_utils.to_categorical(y_val_ns, 6) 

    ## PARAMETERS
    loss_ = 'binary_crossentropy'
    optimizer_ = SGD(lr=0.1)
    epochs_ = 1
    batch_size_ = 100

    print('AE1')
    ### AUTOENCODER 1
    input_img1 = Input(shape=(2048,))
    encoded1 = Dense(40, activation=activation)(input_img1)
    decoded1 = Dense(2048, activation='sigmoid')(encoded1)
    autoencoder1 = Model(input=input_img1, output=decoded1)
    encoder1 = Model(input=input_img1, output=encoded1)
    autoencoder1.compile(optimizer=optimizer_, loss=loss_)
    autoencoder1.fit(X_train_ns, X_train_ns, nb_epoch=epochs_, batch_size=batch_size_, 
                     shuffle=True, validation_data=(X_val_ns, X_val_ns))
    print('AE2')
    ### AUTOENCODER 2
    X_train_enc_ns = encoder1.predict(X_train_ns) 
    X_val_enc_ns = encoder1.predict(X_val_ns)

    input_img2 = Input(shape=(40,))
    encoded2 = Dense(20, activation=activation)(input_img2)
    decoded2 = Dense(40, activation='sigmoid')(encoded2)
    autoencoder2 = Model(input=input_img2, output=decoded2)
    encoder2 = Model(input=input_img2, output=encoded2)
    autoencoder2.compile(optimizer=optimizer_, loss=loss_)
    autoencoder2.fit(X_train_enc_ns, X_train_enc_ns, nb_epoch=epochs_, batch_size=batch_size_,
                     shuffle=True, validation_data=(X_val_enc_ns, X_val_enc_ns))

    print('MLP')
    ### FINE TUNNING
    model = Sequential()
    model.add( Dense(40, activation=activation, input_shape=(2048,)) )
    model.layers[-1].set_weights( autoencoder1.layers[1].get_weights() )
    model.add( Dense(20, activation=activation) )
    model.layers[-1].set_weights( autoencoder2.layers[1].get_weights() )
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer=optimizer_, loss='binary_crossentropy', metrics=['accuracy'])
    # saving net before fine tunning
    model.fit(X_train_s, Y_train_s, nb_epoch=epochs_, batch_size=batch_size_, shuffle=True, validation_data=(X_val_s, Y_val_s))
    # saving results
    acc = model.evaluate(X_test, Y_test, verbose=0)
    acc_list.append(acc)
    pickle.dump(acc_list, open('mlp_acc_AE_{0}'.format(activation),'wb'))
