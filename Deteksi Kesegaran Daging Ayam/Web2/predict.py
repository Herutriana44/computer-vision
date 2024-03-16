"""
File ini berisi class predictor untuk melakukan prediksi menggunakan algoritman KNN dan CNN
dengan mengembalikan label dan probabilitasi hasil prediksi dari kedua algoritma tersebut
"""

import pickle
import tensorflow as tf

class Predictor:
    def __init__(self):
        self.knn_model = pickle.load(open("KNN.pkl", "rb"))
        # self.cnn_model = tf.keras.models.load_model('CNN.h5')

    def predict(self, image):
        # coding untuk melakukan prediksi gambar dengan KNN dan CNN
        # prediksi KNN
        knn_label = self.knn_model.predict(image)
        knn_prob = self.knn_model.predict_proba(image)
        # prediksi CNN
        # cnn_label = self.cnn_model.predict(image)
        # cnn_prob = self.cnn_model.predict_proba(image)

        return knn_label, knn_prob#, cnn_label, cnn_prob
