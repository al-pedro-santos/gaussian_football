import cv2
from pathlib import Path
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

class VideoVGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                            ])


        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        self.features = backbone.features # remove classificador
        self.avgpool = backbone.avgpool
        self.eval()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x
    
    def preprocess_and_save(self, vid_path):
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_features_no_top_2')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        if os.path.exists(out_vid_path):
            x = np.load(out_vid_path, mmap_mode='r')
            return x, None, out_vid_path

        vidcap = cv2.VideoCapture(vid_path)
        features = []
        success = True
        with torch.no_grad():
            while success:
                success, image = vidcap.read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(image)
                    tensor = tensor.unsqueeze(0)

                    feature = self.forward(tensor) # extrai feature
                    feature = feature.squeeze(0).cpu().numpy()
                    features.append(feature)

        vidcap.release()
        features = np.array(features)
        np.save(out_vid_path, features)
        return None, None, out_vid_path
    

    def preprocess(self, video):
        # nessa função não é salvo
        features = []
        success = True
        for i in range(video.shape[0]):
            image = video[i]

            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = self.transform(image)
                tensor = tensor.unsqueeze(0)

                feature = self.forward(tensor) # extrai feature
                feature = feature.squeeze(0).cpu().numpy()
                features.append(feature)

        features = np.array(features)
        return features, None, None
    
    def preprocess_mod(self, vid_path):
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_features_no_top_2')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        if os.path.exists(out_vid_path):
            x = np.load(out_vid_path, mmap_mode='r')
            return x, None, out_vid_path

        vidcap = cv2.VideoCapture(vid_path)
        features = []
        success = True
        with torch.nn_grad():
            while success:
                success, image = vidcap.read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(image)
                    tensor = tensor.unsqueeze(0)

                    feature = self.forward(tensor) # extrai feature
                    feature = feature.squeeze(0).cpu().numpy()
                    features.append(feature)

        vidcap.release()
        features = np.array(features)
        np.save(out_vid_path, features)
        return features, None, out_vid_path