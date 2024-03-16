import onnx
import onnxmltools
import pickle
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from tensorflow import keras


def convertH5(model_file,name_model):
    with open(model_file, 'rb') as f:
        sklearn_model = pickle.load(f)
    initial_type = [('input', FloatTensorType([None, sklearn_model.coef_.shape[1]]))]
    onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)

    onnx_model = onnx.load('model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('model.tf')

    loaded_model = keras.models.load_model('model.tf')
    loaded_model.save(f'{name_model}.h5')