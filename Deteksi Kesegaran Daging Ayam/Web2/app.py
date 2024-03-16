# Import library yang diperlukan
from flask import Flask, render_template, request, redirect, url_for
import os
from hsv import HsvExtractor
from transformation import Transformer
from predict import Predictor
from showImage import ImageShower
import joblib
import numpy as np
from PIL import Image
import random as rd
import cv2
import tensorflow as tf

# Inisiasi Flask
app = Flask(__name__)

# Konfigurasi folder upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Inisiasi objek untuk melakukan ekstraksi fitur pada citra
extractor = HsvExtractor()

# Inisiasi objek untuk melakukan transformasi citra
transformer = Transformer()

# Inisiasi objek untuk melakukan prediksi kelas pada citra
predictor = Predictor()

# Inisiasi objek untuk menampilkan citra
shower = ImageShower()

# Definisikan route '/' yang akan menampilkan halaman index.html
@app.route('/')
def index():
    return render_template('index.html')

# Definisikan route '/upload' yang akan menerima request POST dengan file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Ambil file yang di-upload oleh user
    file = request.files['file']
    filename = file.filename

    # Simpan file tersebut ke dalam folder 'static/uploads'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Redirect ke halaman uploaded_file dengan parameter filename
    return redirect(url_for('uploaded_file', filename=filename))

# Definisikan route '/uploads/<filename>' yang akan menampilkan hasil pengolahan citra
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Load model KNN dan CNN dari file pickle
    knn_model = joblib.load('KNN.pkl')
    # cnn_model = tf.keras.models.load_model('CNN.h5')

    # Transformasi citra dengan HSV
    transformer = Transformer()
    hsv_image = transformer.transform('static/uploads/'+filename)

    # Ekstraksi fitur citra dengan model HsvExtractor
    feature_extractor = HsvExtractor()
    mean_values = feature_extractor.extract('static/uploads/'+filename)

    # Hitung histogram warna HSV
    # hsv_img = cv2.imread(hsv_image)
    # hsv_img = cv2.resize(hsv_img, (100, 100))
    # hsv_hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 2, 2], [0, 180, 0, 256, 0, 256])
    # hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()
    feature_input = np.array([mean_values[3]/255, mean_values[4]/255, mean_values[5]/255])

    # Prediksi kelas menggunakan model KNN dan CNN
    # knn_prediction = knn_model.predict(hsv_hist.reshape(1, -1))
    # knn_prob = knn_model.predict_proba(hsv_hist.reshape(1, -1))[0]
    knn_prediction = knn_model.predict(feature_input.reshape(1, -1))
    knn_prob = knn_model.predict_proba(feature_input.reshape(1, -1))[0]

    # cnn_prediction = cnn_model.predict(hsv_hist.reshape(1, -1))
    label_map = {0: 'Ayam Tidak Segar', 1: 'Ayam Segar'}
    # cnn_label = round(max(cnn_prediction[0]))
    # label = label_map[cnn_label]
    # cnn_prob = cnn_prediction[0]
    result = {
        'knn_prediction': label_map[knn_prediction[0]],
        # 'cnn_prediction': label,
        'knn_prob': knn_prob
        # 'cnn_prob': cnn_prob
    }

    # Hapus static/uploads pada hsv_image
    hsv_image = hsv_image.replace('static/uploads/', '')

    # mean_values to dict
    mean_values = {
        'mean_r': mean_values[0],
        'mean_g': mean_values[1],
        'mean_b': mean_values[2],
        'mean_h': mean_values[3],
        'mean_s': mean_values[4],
        'mean_v': mean_values[5]
        }

    """
    Pada bagian ini, kita mengubah list mean_values menjadi dictionary yang berisi rata-rata nilai untuk setiap channel warna
    Kemudian hasil tersebut akan di-passing sebagai parameter 'mean_values' pada render_template, sehingga nantinya bisa ditampilkan pada halaman web yang menggunakan template uploaded.html
    """
    return render_template('uploaded.html', filename=filename, result=result, hsv_image=hsv_image,mean_values=mean_values)

if __name__ == '__main__':
    app.run(debug=True)