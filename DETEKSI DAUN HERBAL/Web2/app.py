# Import library yang diperlukan
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import joblib
import numpy as np
from PIL import Image
import random as rd
import cv2
import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.neighbors import KNeighborsClassifier
import pickle
import tensorflow as tf
import json
from skimage.measure import label as lab
import uuid

def reverse_dict(d):
    return {v: k for k, v in d.items()}

def crop_object(img):
    blurred = cv2.blur(img, (5,5))
    canny = cv2.Canny(blurred, 50, 200)

    ## find the non-zero min-max coords of canny
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = img[y1:y2, x1:x2]
    return cropped
def normalize(data, min_value, max_value):
    """
    Normalizes a float to the range [0, 1].

    Args:
        data: The float to be normalized.
        min_value: The minimum value in the input data.
        max_value: The maximum value in the input data.

    Returns:
        A normalized float.
    """

    # Check for invalid input

    if not isinstance(data, float):
        raise ValueError("data must be a float")

    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    # Calculate the range of the input data

    range_value = max_value - min_value

    # Normalize the input data

    normalized_data = (data - min_value) / range_value

    return normalized_data

def extract_features(image):
    features = []

    # Mengubah gambar menjadi grayscale
    gray_image = rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)

    # Metric: Jumlah area piksel yang bernilai 1
    labeled_image = label(gray_image > 0)
    props = regionprops(labeled_image)
    area = props[0].area if props else 0

    # Eccentricity: Eksentrisitas region
    if props:
        eccentricity = props[0].eccentricity
    else:
        eccentricity = 0

    # # Gabungkan Metric dan Eccentricity menjadi satu fitur
    # combined_feature = [
    #     normalize(area, 86070.0, 3288384.0), 
    #     normalize(eccentricity, 0.052746550212344645, 0.9687462317467702)]
    # features.extend(combined_feature)

    # GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    # Ekstraksi fitur entropy dan homogenitas
    entropy = -np.sum(glcm * np.log(glcm + np.finfo(float).eps))
    homogeneity = np.sum(glcm / (1.0 + np.abs(0 - 1)))

     # Normalisasi fitur
    normalized_area = normalize(area, 86070.0, 3288384.0)
    normalized_eccentricity = normalize(eccentricity, 0.052746550212344645, 0.9687462317467702)
    normalized_contrast = normalize(contrast, 3.09712535728869, 379.15949047118636)
    normalized_energy = normalize(energy, 0.0303806915008107, 0.6785592519223528)
    normalized_entropy = normalize(entropy, 0, 7)  # Rentang nilai entropy biasanya [0, 7]
    normalized_homogeneity = normalize(homogeneity, 0, 1)  # Rentang nilai homogenitas biasanya [0, 1]

    # Gabungkan semua fitur menjadi satu
    combined_feature = [
        normalized_area, normalized_eccentricity,
        normalized_contrast, normalized_energy,
        normalized_entropy, normalized_homogeneity
    ]

    features.extend(combined_feature)
    

    # features = [i / 255 for i in features]
    # print(features)

    return features

# Fungsi untuk melakukan prediksi dengan model KNN
def predict_with_knn(input_features, model_filename, label_mapping_filename):
    # Memuat model KNN dari file pickle
    with open(model_filename, 'rb') as model_file:
        knn_model = pickle.load(model_file)

    # Memuat label mapping dari file JSON
    with open(label_mapping_filename, 'r') as label_mapping_file:
        label_mapping = json.load(label_mapping_file)

    # Melakukan prediksi
    predicted_label_idx = knn_model.predict([input_features])[0]
    predicted_prob = np.max(knn_model.predict_proba([input_features])[0])

    return predicted_label_idx, predicted_prob


# # Fungsi untuk melakukan prediksi menggunakan model KNN yang sudah di-export
# def predict_with_knn(image, knn_model):
#     # Menggunakan model KNN untuk melakukan prediksi
#     features = extract_features(image)
#     prediction = knn_model.predict([features])
#     return prediction

# def predict_proba_with_knn(image, knn_model):
#     # Menggunakan model KNN untuk melakukan prediksi
#     features = extract_features(image)
#     prediction = knn_model.predict_proba([features])
#     return prediction

# Inisiasi Flask
app = Flask(__name__)

# Konfigurasi folder upload
app.config['UPLOAD_FOLDER'] = os.path.join('static','uploads')

# Definisikan route '/' yang akan menampilkan halaman index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_img', methods=['POST'])
def upload_img():
    # Ambil file yang di-upload oleh user
    file = request.files['file']
    filename = file.filename

    # Simpan file tersebut ke dalam folder 'static/uploads'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Redirect ke halaman uploaded_file dengan parameter filename
    return redirect(url_for('uploaded_file', filename=filename))

# Definisikan route '/upload' yang akan menerima request POST dengan file upload
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Ambil data gambar dari permintaan JSON
        data = request.get_json()
        gambar_data_url = data.get('file')

        if not gambar_data_url:
            return jsonify({'error': 'Data gambar tidak ditemukan', 'success': False})

        # Buat nama unik untuk gambar
        unique_filename = str(uuid.uuid4()) + ".png"

        # Simpan data gambar sebagai berkas di folder '/static/uploads'
        with open(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename), 'wb') as file:
            file.write(base64.b64decode(gambar_data_url.split(",")[1]))

        # Kirim respons berhasil ke klien
        return jsonify({'success': True, 'filename': unique_filename})

    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

# Definisikan route '/uploads/<filename>' yang akan menampilkan hasil pengolahan citra
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    
    # Load model KNN dan CNN dari file pickle
    knn_model = joblib.load('KNN.pkl')

    # Hitung histogram warna HSV
    hsv_img = cv2.imread(os.path.join('static','uploads',filename))
    # hsv_img = crop_object(img)
    features = extract_features(hsv_img)
    Eccentricity_metric = predict_with_knn(features[:2], 'Eccentricity Metric.pkl', 'label_mapping_Eccentricity.json')
    glcm = predict_with_knn(features[2:], 'GLCM.pkl', 'label_mapping_GLCM.json')
    features = [features[:2], features[2:]]
    # knn_prediction = predict_with_knn(hsv_img, knn_model)
    # knn_prob = predict_proba_with_knn(hsv_img, knn_model)

    # with open('label_mapping.json', 'r') as file_json:
    #     label_map = json.load(file_json)
    # cnn_prediction = cnn_model.predict(hsv_hist.reshape(1, -1))
    # features = extract_features(hsv_img)
    # print(label_map)
    # label_map = label_map
    # print(label_map)
    all_features = [
        'Metric',
        'Eccentricity',
        'Contrast',
        'Energy',
        'Entropy',
        'Homogeneity'
    ]

    all_features = {
        1 : all_features[:2], 
        2 : all_features[2:]
        }

    result = {
        'Eccentricity_metric': Eccentricity_metric,
        'GLCM' : glcm,
        # 'cnn_prediction': label,
        # 'knn_prob': np.max(knn_prob)
        # 'cnn_prob': cnn_prob
    }

    print(result)

    features_name = {
        1 : 'Eccentricity_metric',
        2 : 'GLCM'
    }

    """
    Pada bagian ini, kita mengubah list mean_values menjadi dictionary yang berisi rata-rata nilai untuk setiap channel warna
    Kemudian hasil tersebut akan di-passing sebagai parameter 'mean_values' pada render_template, sehingga nantinya bisa ditampilkan pada halaman web yang menggunakan template uploaded.html
    """
    return render_template('uploaded.html', filename=filename, result=result, features=features, features_name=features_name, all_features=all_features)

if __name__ == '__main__':
    app.run(debug=True)