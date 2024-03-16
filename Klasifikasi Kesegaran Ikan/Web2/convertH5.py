import pickle
from keras.models import load_model


def convertH5(model_file,name_model):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Simpan model dalam format .h5
    model.save(f'{name_model}.h5')

convertH5('CNN.pkl', 'CNN')