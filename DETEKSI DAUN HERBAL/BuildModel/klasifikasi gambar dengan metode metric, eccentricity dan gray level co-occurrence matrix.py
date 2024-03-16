import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.measure import regionprops_table
from skimage.measure import moments_hu

# Load dataset (contoh dengan dataset digits)
data = load_digits()
X = data.images
y = data.target

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ekstraksi fitur menggunakan metode-metode yang disebutkan
def extract_features(image):
    features = []
    
    # Metric: jumlah area piksel yang bernilai 1
    label_image = label(image > 0)
    props = regionprops(label_image)
    area = props[0].area if props else 0
    features.append(area)
    
    # Eccentricity: eksentrisitas region
    if props:
        eccentricity = props[0].eccentricity
    else:
        eccentricity = 0
    features.append(eccentricity)
    
    # GLCM (Gray Level Co-occurrence Matrix)
    glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    features.append(contrast)
    features.append(energy)
    
    return features

# Ekstraksi fitur dari semua gambar
X_train_features = np.array([extract_features(image) for image in X_train])
X_test_features = np.array([extract_features(image) for image in X_test])

# Klasifikasi dengan Support Vector Machine (SVM)
clf = SVC()
clf.fit(X_train_features, y_train)

# Prediksi
y_pred = clf.predict(X_test_features)

# Evaluasi hasil
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
