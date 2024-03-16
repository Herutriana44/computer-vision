"""
Ini adalah file yang berisi class untuk mengestrak gambar dari RGB ke HSV
dan mengembalikan rata-rata nilai Red, Green, Blue (RGB) Hue, Saturation, Value (HSV)
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

class HsvExtractor:
    def __init__(self):
        pass
    
    def extract(self, image):
        # pastikan ukuran gambar adalah (n, m, 3)
        image = plt.imread(image)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        elif image.shape[-1] != 3:
            image = image[:,:,:3]
        # konversi gambar dari RGB ke HSV
        hsv_image = matplotlib.colors.rgb_to_hsv(image)
        
        # ekstraksi nilai HSV
        h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
        
        # hitung nilai rata-rata dari masing-masing channel
        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        
        mean_r = np.mean(image[:,:,0])
        mean_g = np.mean(image[:,:,1])
        mean_b = np.mean(image[:,:,2])

        return mean_r, mean_g, mean_b, mean_h, mean_s, mean_v
