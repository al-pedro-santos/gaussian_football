import cv2

class ImageResizePreprocessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interp = interpolation

    def resize_prepocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.interp)
