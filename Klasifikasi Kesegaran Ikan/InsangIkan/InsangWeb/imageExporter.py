# file berisi class untuk ekspor gambar dari RGB ke HSV

import matplotlib.pyplot as plt
import numpy as np

class ImageExporter:
    def __init__(self):
        pass
    
    def export(self, hsv_image, hsv_image_name):
        # normalisasi nilai floating-point pada gambar
        hsv_image_norm = hsv_image.astype(np.float32) / np.max(hsv_image)
        # simpan gambar ke file
        plt.imsave(hsv_image_name, hsv_image_norm)
        return hsv_image_name
