from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision

from datasets_mel_video import default_mel_transform

train_video_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

class MultiModalDataset(Dataset):
    def __init__(
        self,
        csv_path,
        pair=False,
        split=None,
        score_col="arousal_score",
        binary_label=False,
        threshold=0.5,
        target_shape=(128, 256),
        is_grayscale=False,
        video_transform=None,
        mel_transform=None,
        dtype=torch.float32,
    ):

        self.df = pd.read_csv(csv_path)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        n_before = len(self.df)

        self.df = self.df[self.df["clip_path"].notna() & self.df["mel_path"].notna()].copy()

        self.df = self.df[
            self.df["clip_path"].apply(lambda p: Path(p).exists())
            & self.df["mel_path"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.df)}/{n_before} exemplos válidos.")

        self.score_col = score_col
        self.binary_label = binary_label
        self.threshold = threshold
        self.pair = pair

        if self.pair:
            self.low_df = self.df[self.df[score_col] < threshold].reset_index(drop=True)
            self.high_df = self.df[self.df[score_col] >= threshold].reset_index(drop=True)

            print(f"Low: {len(self.low_df)}")
            print(f"High: {len(self.high_df)}")

        self.is_grayscale = is_grayscale
        self.video_transform = video_transform
        self.mel_transform = (mel_transform if mel_transform is not None else default_mel_transform(target_shape))
        self.dtype = dtype

    def __len__(self):
        if self.pair:
            return max(len(self.low_df), len(self.high_df)) # é balanceado, entãoé esperado que sejam de mesmo tamanho
        return len(self.df)

    def _load_sample(self, row):
        # video
        video, _, _ = torchvision.io.read_video(row["clip_path"], pts_unit="sec")
        video = video.to(self.dtype) / 255.
        video = video.permute(0, 3, 1, 2)

        if self.is_grayscale and video.shape[1] == 3:
            video = video.mean(dim=1, keepdim=True)

        if self.video_transform is not None:
            video = self.video_transform(video)
        
        mask = torch.ones(video.shape[0], dtype=torch.bool)

        # mel
        mel = np.load(row["mel_path"])

        mel_min = mel.min()
        mel_max = mel.max()

        if mel_max > mel_min:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

        mel = torch.tensor(mel, dtype=self.dtype).unsqueeze(0)

        if self.mel_transform is not None:
            mel = self.mel_transform(mel)

        # label

        score = float(row[self.score_col])

        if self.binary_label:
            score = float(score >= self.threshold)

        score = torch.tensor(score, dtype=self.dtype)
        return video, mask, mel, score
    

    def __getitem__(self, idx):
        if not self.pair:
            row = self.df.iloc[idx]
            return self._load_sample(row)

        # modo pareado:
        low_row = self.low_df.iloc[idx % len(self.low_df)]
        high_row = self.high_df.iloc[np.random.randint(len(self.high_df))]

        low_sample = self._load_sample(low_row)
        high_sample = self._load_sample(high_row)

        return low_sample, high_sample # video_low, mask_low, mel_low, score_low, video_high, mask_high, mel_high, score_high

    @property
    def scores(self):

        vals = self.df[self.score_col].values.astype(float)

        if self.binary_label:
            vals = (vals >= self.threshold).astype(float)

        return vals
    

def multimodal_collate_fn(batch):
    videos, masks, mels, labels = zip(*batch)
    max_frames = max(v.shape[0] for v in videos)

    padded_videos = []
    padded_masks = []

    for video, mask in zip(videos, masks):
        T = video.shape[0]
        if T < max_frames:
            pad_video = torch.zeros((max_frames - T, *video.shape[1:]), dtype=video.dtype,)
            pad_mask = torch.zeros(max_frames - T, dtype=torch.bool,)

            video = torch.cat([video, pad_video], dim=0)
            mask = torch.cat([mask, pad_mask], dim=0)

        padded_videos.append(video)
        padded_masks.append(mask)

    return (
        torch.stack(padded_videos),
        torch.stack(padded_masks),
        torch.stack(mels),
        torch.stack(labels),
    )


def multimodal_pair_collate_fn(batch):
    low_batch = [item[0] for item in batch]
    high_batch = [item[1] for item in batch]

    low = multimodal_collate_fn(low_batch)
    high = multimodal_collate_fn(high_batch)

    return low, high


def build_multimodal_dataloader(
    csv_path,
    split,
    pair,
    batch_size,
    shuffle,
    num_workers,
    score_col="arousal_score",
    binary_label=False,
    threshold=0.5,
    target_shape=(128, 256),
    is_grayscale=False,
    video_transform=None,
    mel_transform=None,
    dtype=torch.float32,
    pin_memory=False,
):

    dataset = MultiModalDataset(
        csv_path=csv_path,
        pair=pair,
        split=split,
        score_col=score_col,
        binary_label=binary_label,
        threshold=threshold,
        target_shape=target_shape,
        is_grayscale=is_grayscale,
        video_transform=video_transform,
        mel_transform=mel_transform,
        dtype=dtype,
    )

    if pair:
        collate_fn = multimodal_pair_collate_fn
    else:
        collate_fn = multimodal_collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

'''
No treinamento:

- se pair = False:
    videos, masks, mels, scores = batch
- se pair = True
    (low, high) = batch

    low_video, low_mask, low_mel, low_score = low
    high_video, high_mask, high_mel, high_score = high

'''