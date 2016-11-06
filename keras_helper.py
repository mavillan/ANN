from keras.models import model_from_json


def save_keras_model(model, model_name):
    model_name = model_name.split('.')[0]
    # serialize model to JSON
    model_json = model.to_json()
    with open('{0}.json'.format(model_name), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('{0}.h5'.format(model_name))


def load_keras_model(model_name):
    model_name = model_name.split('.')[0]
    # load json and create model
    json_file = open('{0}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('{0}.h5'.format(model_name))
    return loaded_model

