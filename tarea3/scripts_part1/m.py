import pandas as pd
import numpy as np
import math
import time
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def save_keras_model(model, model_name):
    model_name = model_name.split('.')[0]
    # serialize model to JSON
    model_json = model.to_json()
    with open('{0}.json'.format(model_name), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('{0}.h5'.format(model_name))

def create_dataset(dataset, lag=1):
    dataX,dataY = [],[]
    for i in range(lag,len(dataset)):
        dataY.append(dataset[i])
        x = []
        for j in range(i-lag,i):
            x.append(dataset[j])
        dataX.append(x)
    return np.array(dataX), np.array(dataY)

if __name__=='__main__':
    path = '/user/a/asalinas/ANN/t3/'
    name = 'outputs/m'

    url = 'http://www.inf.utfsm.cl/~cvalle/international-airline-passengers.csv'
    dataframe = pd.read_csv(url, sep=',', usecols=[1], engine='python', skipfooter=3)
    dataframe[:] = dataframe[:].astype('float32')
    df_train, df_test = dataframe[0:96].values, dataframe[96:].values

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train)
    stream_train_scaled = scaler.transform(df_train)
    stream_test_scaled = scaler.transform(df_test)
    
    lag = 3
    trainX, trainY = create_dataset(stream_train_scaled, lag)
    testX, testY = create_dataset(stream_test_scaled, lag)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    batch_size = 1
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, lag, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, lag, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    start_time = time.time()
    for i in range(100):
        model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    elapsed_time = time.time() - start_time
    save_keras_model(model, path+name)

    trainPredict = model.predict(trainX, batch_size=batch_size)
    trainPredict = scaler.inverse_transform(trainPredict)

    testPredict = model.predict(testX, batch_size=batch_size)
    testPredict = scaler.inverse_transform(testPredict)

    arch = open(path+name+'.txt','w')
    arch.write(str(elapsed_time)+'\n')
    for i in trainPredict:
        arch.write(str(i[0])+' ')
    arch.write('\n')
    for i in testPredict:
        arch.write(str(i[0])+' ')
    arch.close()