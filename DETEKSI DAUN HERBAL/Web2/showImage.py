from PIL import Image
from io import BytesIO
import base64

class ImageShower:
    def __init__(self):
        pass
    
    def show(self, image):
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
