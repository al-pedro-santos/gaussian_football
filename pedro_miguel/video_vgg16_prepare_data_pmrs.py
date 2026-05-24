# import the necessary packages
import cv2
from pathlib import Path
import os
import torch
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights


class VideoVGG16PrepareData:
    def __init__(self, verbose=500):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.verbose = verbose

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*list(base.children())[:-1])
        self.feature_extractor.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)

    def preprocess_and_save(self, vid_path, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_data_preparation')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.pt')

        if os.path.exists(out_vid_path):
            return None, None, out_vid_path
        count = 0
        # print('[INFO] Extracting frames from video: ', vid_path)
        vidcap = cv2.VideoCapture(vid_path)
        frames = []
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
            count += 1

        features = self._run_vgg(torch.stack(frames))
        torch.save(features, out_vid_path)
        return features, None, out_vid_path

    def preprocess(self, vid_path, video, *_):
        frames = []
        for i in range(video.shape[0]):
            frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))

        features = self._run_vgg(torch.stack(frames))
        return features, None, None

    def _run_vgg(self, frames):
        with torch.no_grad():
            frames   = frames.to(self.device)
            features = self.feature_extractor(frames)        # (T, 512, 7, 7)
            features = features.view(features.size(0), -1)  # (T, 25088)
        return features.cpu()