"""
File ini bersisi class transformer untuk melakukan transformasi dari RGB ke HSV
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random as rd
from imageExporter import ImageExporter

class Transformer:
    def __init__(self):
        pass
    
    def transform(self, image):
        image = plt.imread(image)
        # pastikan ukuran gambar adalah (n, m, 3)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        elif image.shape[-1] != 3:
            image = image[:,:,:3]
        # konversi gambar dari RGB ke HSV
        hsv_image = matplotlib.colors.rgb_to_hsv(image)
        # eksport gambar
        hsv_image_name = 'static/uploads/hsv_img/hsv_image' + str(rd.randint(0, 100000)) + '.jpg'
        exporter = ImageExporter()
        hsv_image_name = exporter.export(hsv_image, hsv_image_name)
        return hsv_image_name
