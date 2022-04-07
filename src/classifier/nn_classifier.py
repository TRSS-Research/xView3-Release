import pickle
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import model_from_json 


def load_model(filepath):
    if filepath.upper().endswith('PKL'):
        with open(filepath, 'rb') as fo:
            return pickle.load(fo)
    else:
        try:
            return keras_load_model(filepath)
        except ValueError: # weights not model
            filename = '.'.join(filepath.split('.')[:-1])
            json_filepath = filename + '.json'
            with open(json_filepath, 'r') as fo:
                model_arch = fo.read()
            model = model_from_json(model_arch)
            model.load_weights(filepath)
            return model
