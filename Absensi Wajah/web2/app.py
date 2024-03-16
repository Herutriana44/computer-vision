from flask import Flask, Response, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, url_for, session
import cv2
from werkzeug.utils import secure_filename
import numpy as np
import base64
from keras.callbacks import LambdaCallback
from io import BytesIO
from functools import wraps
from PIL import Image
from keras.models import load_model
import joblib
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import datetime
import os
from keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_logged_in' not in session or not session['user_logged_in']:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def augment_image(image_path, output_dir, num_augmented_images=20):
    # Create an ImageDataGenerator object
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Read the input image
    img = plt.imread(image_path)
    # Change the dimensions of the image to (1, x, y, channel)
    img = img.reshape((1,) + img.shape)
    # Create a new file name
    label = image_path.split('/')[-1].split('.')[0]
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=output_dir, save_prefix=label, save_format='jpg'):
        i += 1
        if i > num_augmented_images:
            break

def build_model():
    # Path to the dataset directory
    data_dir = 'static/dataset'

    # List all image files in the dataset directory
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    # Load images and extract features
    X = []
    y = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        X.append(img)
        label = image_file.split('/')[-1].split('_')[0]
        y.append(label)

    # Convert data to numpy arrays
    X = np.array(X, dtype=np.int16)
    y = np.array(y, dtype=np.int16)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    # Compile the CNN model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train_encoded, num_classes=len(np.unique(y)))
    y_test_encoded = encoder.transform(y_test)
    y_test = to_categorical(y_test_encoded, num_classes=len(np.unique(y)))

    def on_epoch_end(epoch, logs):
        yield f"Epoch {epoch + 1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}\n"

    epoch_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # Train the CNN model
    model.fit(X_train, y_train, epochs=15, batch_size=32, callbacks=[epoch_callback])

    # Save the CNN model to file
    #pickle.dump(model, open('./CNN.h5', 'wb'))
    model.save('model.h5')
    
    # Export model to JSON file
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def crop_face(image_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cropped_image = image[y:y+h, x:x+w]
                cv2.imwrite(image_path, cropped_image)
        else:
            print("No face detected")

def create_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="absensi_wajah"
    )
    return connection

def get_all_pegawai():
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT id, nama FROM pegawai"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    pegawai_list = []
    for result in results:
        pegawai_list.append({'id': result[0], 'nama': result[1]})

    return pegawai_list


def hapus_pegawai(pegawai_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "DELETE FROM pegawai WHERE id = %s"
    cursor.execute(query, (pegawai_id,))
    conn.commit()
    cursor.close()
    conn.close()

    for filename in os.listdir("static/dataset/"):
        # cek apakah file memiliki ekstensi .jpg dan awalan prefix yang diinputkan
        if filename.endswith(".jpg") and filename.startswith(f"{pegawai_id}"):
            # hapus file
            os.remove(filename)
            print(f"File {filename} berhasil dihapus.")

    retrain_model()

def save_absen(id_pegawai, tipe_absen):
    conn = create_connection()
    cursor = conn.cursor()
    query = "INSERT INTO absen (id_pegawai, waktu_absen, tipe_absen) VALUES (%s, NOW(), %s)"
    cursor.execute(query, (id_pegawai, tipe_absen))
    conn.commit()
    cursor.close()
    conn.close()

def get_admin(username):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT id, username, password FROM admin WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        return {'id': result[0], 'username': result[1], 'password': result[2]}
    else:
        return None


app = Flask(__name__)
app.secret_key = 'ABSENSI'


app.config['UPLOAD_FOLDER'] = 'static/image'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

# Load pre-trained CNN model
model = load_model('model.h5')

@app.route('/pegawai/add', methods=['POST'])
def add_pegawai():
    id = request.form['id']
    nama = request.form['nama']
    file = request.files['gambar']
    if file and allowed_file(file.filename):
        file_ext = os.path.splitext(file.filename)[1] # Mendapatkan ekstensi file
        filename = id + file_ext # Menggabungkan ID dengan ekstensi file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        crop_face(file_path)
        augment_image(file_path, 'static/dataset')

        # Latih ulang model dengan data baru
        retrain_model()

        # Tambahkan pegawai baru ke database
        conn = create_connection()
        cursor = conn.cursor()
        query = "INSERT INTO pegawai (id, nama, file_gambar) VALUES (%s, %s, %s)"
        cursor.execute(query, (id, nama, file_path))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': 'File not allowed'})



def predict_image(image, model):
    img = cv2.resize(image, (150, 150), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert image to RGB format
    img = np.array([img], dtype=np.int16)  # Add an extra dimension to the image
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_score = prediction[0][predicted_label]
    return predicted_label, predicted_score


def detect_and_crop_face(img):
    img = cv2.convertScaleAbs(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = img[y:y+h, x:x+w]
        break
    return img, faces

def draw_faces_on_image(img, faces, result):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['image']
    img_data = base64.b64decode(data.split(',')[1])
    image = Image.open(BytesIO(img_data))
    image = np.array(image, dtype=np.int16)
    cropped_face, Faces = detect_and_crop_face(image)
    predicted_label, predicted_score = predict_image(cropped_face, model)
    
    # Save absen to database
    current_time = datetime.datetime.now().time()
    if current_time < datetime.time(12, 0):
        tipe_absen = "pagi"
    elif current_time < datetime.time(15, 0):
        tipe_absen = "siang"
    else:
        tipe_absen = "sore"
    save_absen(int(predicted_label), tipe_absen)
    
    # Get the employee name
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT nama FROM pegawai WHERE id = %s"
    cursor.execute(query, (int(predicted_label),))
    employee_name = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    result = "Nama: {}, Predict Score: {:.2f}, Absen: {}".format(employee_name, predicted_score, tipe_absen)

    image_with_faces = draw_faces_on_image(image, Faces, result)

    # Draw the prediction text on the cropped face image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 0, 0)
    thickness = 1
    cv2.putText(image_with_faces, result, (10, 20), font, scale, color, thickness, cv2.LINE_AA)

    # Encode the image with the prediction text as a base64 string
    retval, buffer = cv2.imencode('.jpg', image_with_faces)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'result': result, 'image': encoded_image})

@app.route('/haarcascade_frontalface_default.xml')
def serve_cascade_file():
    return send_from_directory('static', 'haarcascade_frontalface_default.xml')

@app.route('/login')
def admin_login():
    return render_template('login.html')

@app.route('/logincek', methods=['GET', 'POST'])
def admin_login_cek():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = get_admin(username)

        if admin and admin['password'] == password:
            # Logika untuk mengatur sesi dan mengarahkan ke halaman dashboard admin
            # Anda perlu mengimpor dan mengatur flask_login jika ingin menggunakan sesi login
            flash('Logged in successfully', 'success')
            session['user_logged_in'] = True
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/pegawai', methods=['GET', 'POST'])
@login_required
def pegawai():
    if request.method == 'POST':
        nama = request.form['nama']
        add_pegawai(nama)
        return redirect(url_for('pegawai'))
    pegawai_list = get_all_pegawai()
    return render_template('pegawai.html', pegawai_list=pegawai_list)

def get_pegawai_by_id(pegawai_id):
    conn = create_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM pegawai WHERE id = %s"
    cursor.execute(query, (pegawai_id,))
    pegawai = cursor.fetchone()
    cursor.close()
    conn.close()
    return pegawai

def update_pegawai_by_id(pegawai_id, new_name):
    conn = create_connection()
    cursor = conn.cursor()
    query = "UPDATE pegawai SET nama = %s WHERE id = %s"
    cursor.execute(query, (new_name, pegawai_id))
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/pegawai/edit/<int:pegawai_id>')
def edit_pegawai(pegawai_id):
    pegawai_data = get_pegawai_by_id(pegawai_id)
    return render_template('edit_pegawai.html', pegawai=pegawai_data, idpegawai=pegawai_id)

@app.route('/pegawai/update/<int:pegawai_id>', methods=['POST','GET'])
def update_pegawai(pegawai_id):
    new_name = request.form['nama']
    update_pegawai_by_id(pegawai_id, new_name)
    return redirect(url_for('pegawai'))


@app.route('/pegawai/hapus/<int:pegawai_id>')
def Hapus_Pegawai(pegawai_id):
    hapus_pegawai(pegawai_id)
    return redirect(url_for('pegawai'))

@app.route('/absen')
def absen():
    filter_date = request.args.get('filter_date', None)

    conn = create_connection()
    cursor = conn.cursor()
    
    query = '''
    SELECT a.id_pegawai, p.nama, DATE(a.waktu_absen) as tanggal, 
           TIME(MIN(a.waktu_absen)) as absen_masuk, TIME(MAX(a.waktu_absen)) as absen_keluar
    FROM absen a
    JOIN pegawai p ON a.id_pegawai = p.id
    WHERE 1
    '''
    
    if filter_date:
        query += " WHERE DATE(a.waktu_absen) = %s"
        query += '''
        GROUP BY a.id_pegawai, DATE(a.waktu_absen)
        ORDER BY a.waktu_absen
        '''
        cursor.execute(query, (filter_date,))
    else:
        query += '''
        GROUP BY a.id_pegawai, DATE(a.waktu_absen)
        ORDER BY a.waktu_absen
        '''
        cursor.execute(query)

    absen_records = cursor.fetchall()
    cursor.close()
    conn.close()

    absen_list = []
    for record in absen_records:
        absen_list.append({
            'id_pegawai': record[0],
            'nama': record[1],
            'tanggal': record[2].strftime("%Y-%m-%d") if isinstance(record[2], datetime.datetime) else "Unknown",
            'absen_masuk': record[3].strftime("%H:%M:%S") if isinstance(record[3], datetime.datetime) else "Unknown",
            'absen_keluar': record[4].strftime("%H:%M:%S") if isinstance(record[4], datetime.datetime) else "Unknown"
        })

    return render_template('absen.html', absen_list=absen_list)

@app.route('/logout')
def admin_logout():
    session.pop('logged_in', None)
    session.pop('user_logged_in', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('absen'))

@app.route("/train")
def train():
    return Response(build_model(), content_type="text/plain")

def retrain_model():
    build_model()


if __name__ == '__main__':
    app.run(debug=True)
