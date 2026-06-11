from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

# =======================================
# Mel spectograma
# =======================================


def default_mel_transform(target_shape=None):
    ops = []

    if target_shape is not None:
        ops.append(transforms.Resize(target_shape))

    ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(ops)


class MelSpectrogramDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split=None,
        score_col="arousal_score",
        binary_label=False,
        threshold=0.5,
        target_shape=(128, 256),
        transform=None,
        dtype=torch.float32,
    ):

        self.df = pd.read_csv(csv_path)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        # tratamento dos mel spectrogramas sem audio
        n_before = len(self.df)

        self.df = self.df[self.df["mel_path"].notna()].copy()
        self.df = self.df[
            self.df["mel_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.df)}/{n_before} exemplos válidos.")

        self.score_col = score_col
        self.binary_label = binary_label
        self.threshold = threshold
        self.transform = transform or default_mel_transform(target_shape)
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])

        # Normalização min-max por amostra para [0, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max > mel_min:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

        # [n_mels, T] -> [1, n_mels, T]
        mel_tensor = torch.tensor(mel, dtype=self.dtype).unsqueeze(0)

        # transform
        if self.transform is not None:
            mel_tensor = self.transform(mel_tensor)

        # score
        score = float(row[self.score_col])
        if self.binary_label:
            score = float(score >= self.threshold)  # classificação 0-1 para highlight
        score = torch.tensor(score, dtype=self.dtype)

        return mel_tensor, score

    @property
    def scores(self):
        # retorna um numpy array com os rótulos (0-1 ou aurosal)
        vals = self.df[self.score_col].values.astype(float)
        if self.binary_label:
            vals = (vals >= self.threshold).astype(float)

        return vals

    def __repr__(self):
        return (
            f"MelSpectrogramDataset("
            f"n_samples={len(self)}, "
            f"score='{self.score_col}', "
            f"binary_label={self.binary_label})"
        )


# =======================================
# Vídeo
# =======================================


class VideoClipDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split=None,
        score_col="arousal_score",
        binary_label=False,
        threshold=0.5,
        is_grayscale=False,
        transform=None,
        dtype=torch.float32,
    ):

        self.df = pd.read_csv(csv_path)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        n_before = len(self.df)

        self.df = self.df[self.df["clip_path"].notna()].copy()
        self.df = self.df[
            self.df["clip_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.df)}/{n_before} exemplos válidos.")

        self.score_col = score_col
        self.binary_label = binary_label
        self.threshold = threshold
        self.is_grayscale = is_grayscale
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video, _, _ = torchvision.io.read_video(row["clip_path"], pts_unit="sec")

        # [T,H,W,C] -> float [0,1]
        video = video.to(self.dtype) / 255.0

        # [T,H,W,C] -> [T,C,H,W]
        video = video.permute(0, 3, 1, 2)

        # correção de bug
        if self.is_grayscale and video.shape[1] == 3:
            video = video.mean(dim=1, keepdim=True)

        expected_channels = 1 if self.is_grayscale else 3

        if video.shape[1] != expected_channels:
            raise ValueError(
                f"Vídeo '{row['clip_path']}' possui "
                f"{video.shape[1]} canais, mas o dataset "
                f"esperava {expected_channels}."
            )

        if self.transform is not None:
            video = torch.stack([self.transform(frame) for frame in video])

        mask = torch.ones(video.shape[0], dtype=torch.bool)

        score = float(row[self.score_col])
        if self.binary_label:
            score = float(score >= self.threshold)
        score = torch.tensor(score, dtype=self.dtype)
        return video, mask, score

    @property
    def scores(self):
        vals = self.df[self.score_col].values.astype(float)
        if self.binary_label:
            vals = (vals >= self.threshold).astype(float)
        return vals

    def __repr__(self):
        return (
            f"VideoClipDataset("
            f"n_samples={len(self)}, "
            f"score='{self.score_col}', "
            f"binary_label={self.binary_label}, "
            f"is_grayscale={self.is_grayscale})"
        )


def video_collate_fn(batch):
    # função para padding
    # os clips das batchs devem ter a mesma quantidade de frames
    videos, masks, labels = zip(*batch)
    max_frames = max(v.shape[0] for v in videos)

    padded_videos = []
    padded_masks = []

    for video, mask in zip(videos, masks):
        T = video.shape[0]
        if T < max_frames:
            pad_video = torch.zeros(
                (max_frames - T, *video.shape[1:]), dtype=video.dtype
            )
            pad_mask = torch.zeros(max_frames - T, dtype=torch.bool)

            video = torch.cat([video, pad_video], dim=0)
            mask = torch.cat(
                [mask, pad_mask], dim=0
            )  # mascara 1 -> frame real, 0 -> frame de padding

        padded_videos.append(video)
        padded_masks.append(mask)

    return (torch.stack(padded_videos), torch.stack(padded_masks), torch.stack(labels))
