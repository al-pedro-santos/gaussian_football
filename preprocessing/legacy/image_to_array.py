# fazendo image to tensor no torch
from PIL import Image
import torchvision
from torchvision import transforms

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        img = Image.open(image)
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img)
        