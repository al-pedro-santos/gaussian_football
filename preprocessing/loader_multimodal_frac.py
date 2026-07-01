from pathlib import Path

import numpy as np
import pandas as pd

import random
from torch.utils.data import Sampler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision
from collections import defaultdict

from datasets_mel_video import default_mel_transform

# Data Augmentation no Mel Espectrograma

class SpecAugment:
    def __init__(self, freq_mask_param=20, time_mask_param=40, n_freq_masks=1, n_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel):
        mel = mel.clone()
        _, H, W = mel.shape

        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(H - f, 0))
            mel[:, f0:f0+f, :] = 0

        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(W - t, 0))
            mel[:, :, t0:t0+t] = 0

        return mel


class AddGaussianNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, mel):
        return mel + torch.randn_like(mel) * self.std

# Utilidades

TARGET_SHAPE = (128, 256)

train_video_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

train_mel_transform = v2.Compose([
    default_mel_transform(TARGET_SHAPE),
    AddGaussianNoise(std=0.05),
    SpecAugment(freq_mask_param=20, time_mask_param=40),
])

split_pt = {
    "train": "Treino",
    "valid": "Validação",
    "test" : "Teste",
}

# Classes Principais
class GroupSampler(Sampler):
    '''
    Fazer shuffle de acordo com os grupos (janelas)

    frac: fração (0 < frac <= 1) dos grupos a usar em cada época.
          Um subconjunto novo é sorteado a cada época (a cada chamada de __iter__).
          frac=1.0 (padrão) mantém o comportamento original.
    '''
    def __init__(self, groups, shuffle=True, frac=1.0):
        if not (0 < frac <= 1.0):
            raise ValueError("frac deve estar em (0, 1].")
        self.groups = groups
        self.shuffle = shuffle
        self.frac = frac
        self._n_groups_epoch = max(1, round(len(self.groups) * self.frac))

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.groups)).tolist()
        else:
            order = list(range(len(self.groups)))

        if self.frac < 1.0:
            order = order[:self._n_groups_epoch]

        for g in order:
            yield from self.groups[g]

    def __len__(self):
        if self.frac < 1.0:
            return sum(len(self.groups[g]) for g in range(self._n_groups_epoch))
        return sum(len(g) for g in self.groups)
    

class PairGroupSampler(Sampler):
    """
    Faz shuffle entre os pares de grupos.
    Exclusivo para a loss margin ranking.

    frac: fração (0 < frac <= 1) dos pares de grupos a usar em cada época.
          Um subconjunto novo é sorteado a cada chamada de __iter__ (ou seja,
          a cada época, já que o DataLoader chama __iter__ uma vez por época).
          frac=1.0 (padrão) mantém o comportamento original, usando todos os pares.
    """

    def __init__(self, group_pairs, shuffle=True, frac=1.0):
        if not (0 < frac <= 1.0):
            raise ValueError("frac deve estar em (0, 1].")
        self.group_pairs = group_pairs
        self.shuffle = shuffle
        self.frac = frac
        self._epoch_len = max(1, round(len(self.group_pairs) * self.frac))

    def __iter__(self):
        order = list(range(len(self.group_pairs)))
        if self.shuffle:
            order = torch.randperm(len(self.group_pairs)).tolist()

        if self.frac < 1.0:
            order = order[:self._epoch_len]

        yield from order

    def __len__(self):
        return self._epoch_len


class MultiModalDataset(Dataset):
    def __init__(
        self,
        csv_path,
        groups=False,
        column_groups_id=None,  # Nome da coluna que identifica os grupos. Amostras com o mesmo valor pertencem ao mesmo grupo e permanecem na ordem original
        pair=False,
        split=None,
        score_col="arousal_score",
        binary_label=False,
        threshold=0.5,
        target_shape=TARGET_SHAPE,
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

        print(f"Dataset de {split_pt[split]}: {len(self.df)}/{n_before} exemplos válidos.")

        self.score_col = score_col
        self.binary_label = binary_label
        self.threshold = threshold
        self.pair = pair

        self.column_groups_id = column_groups_id
        self.groups = None
        self.group_pairs = None

        if groups:
            if column_groups_id is None:
                raise ValueError("column_groups_id deve ser informado quando groups=True.")
            
            grouped = defaultdict(list)

            for idx, row in self.df.iterrows():
                grouped[row[column_groups_id]].append(idx)

            self.groups = list(grouped.values())

            print(f"{len(self.groups)} grupos encontrados.")

        self.is_grayscale = is_grayscale
        self.video_transform = video_transform
        self.mel_transform = (mel_transform if mel_transform is not None else default_mel_transform(target_shape))
        self.dtype = dtype

        if self.pair:
            self.low_df = self.df[self.df[score_col] < threshold].reset_index(drop=True)
            self.high_df = self.df[self.df[score_col] >= threshold].reset_index(drop=True)

            print(f"Low: {len(self.low_df)}")
            print(f"High: {len(self.high_df)}\n")

            # Caso não existam grupos, pareamento antigo
            if not groups:
                print("Pareamento aleatório.\n")
                return

            low_groups = defaultdict(list)
            for idx, row in self.low_df.iterrows():
                low_groups[row[column_groups_id]].append(idx)

            high_groups = defaultdict(list)
            for idx, row in self.high_df.iterrows():
                high_groups[row[column_groups_id]].append(idx)

            # organiza por tamanho para compatibilidade entre as janelas
            low_by_size = defaultdict(list)

            for g in low_groups.values():
                low_by_size[len(g)].append(g)

            high_by_size = defaultdict(list)

            for g in high_groups.values():
                high_by_size[len(g)].append(g)

            # pareamento apenas de grupos de mesmo tamanho para o margin ranking
            self.group_pairs = []

            for size in low_by_size:
                if size not in high_by_size:
                    continue
                lows = low_by_size[size]
                highs = high_by_size[size]
                n = min(len(lows), len(highs))
                for i in range(n):
                    self.group_pairs.append((lows[i], highs[i]))
            print(f"{len(self.group_pairs)} pares de grupos criados.\n")

    def __len__(self):
        if self.pair:
            if self.groups is not None:
                return len(self.group_pairs)
            return max(len(self.low_df), len(self.high_df))

        if self.groups is not None:
            return len(self.df)  # o sampler controla a ordem

        return len(self.df)

    def _load_sample(self, row, retry=True):
        """
        Carrega uma amostra (vídeo + mel + label).
        Se o vídeo estiver corrompido, tenta recuperar ou retorna None.
        """
        try:
            # video
            video, _, _ = torchvision.io.read_video(row["clip_path"], pts_unit="sec")
            video = video.to(self.dtype) / 255.
            video = video.permute(0, 3, 1, 2)

            if self.is_grayscale and video.shape[1] == 3:
                video = video.mean(dim=1, keepdim=True)

            if self.video_transform is not None:
                video = self.video_transform(video)
            
            mask = torch.ones(video.shape[0], dtype=torch.bool)

        except Exception as e:
            # Vídeo corrompido — retorna None para re-amostrar
            return None

        # mel
        try:
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

        except Exception as e:
            # Mel corrompido — retorna None para re-amostrar
            return None

        # label
        score = float(row[self.score_col])

        if self.binary_label:
            score = float(score >= self.threshold)

        score = torch.tensor(score, dtype=self.dtype)
        return video, mask, mel, score
    

    def __getitem__(self, idx):
        if not self.pair:
            # Tenta carregar até conseguir um exemplo válido
            for attempt in range(10):
                try_idx = idx if attempt == 0 else np.random.randint(len(self.df))
                row = self.df.iloc[try_idx]
                sample = self._load_sample(row)
                if sample is not None:
                    return sample
            # Se falhar 10 vezes, retorna dummy
            raise RuntimeError(f"Não conseguiu carregar amostra válida após 10 tentativas")

        # pareamento simples
        if self.groups is None:
            for attempt in range(10):
                low_row = self.low_df.iloc[np.random.randint(len(self.low_df))]
                high_row = self.high_df.iloc[np.random.randint(len(self.high_df))]

                low_sample = self._load_sample(low_row)
                high_sample = self._load_sample(high_row)
                
                if low_sample is not None and high_sample is not None:
                    return (low_sample, high_sample)
            
            raise RuntimeError(f"Não conseguiu carregar par válido após 10 tentativas")

        # dataset pareado por grupos
        low_group, high_group = self.group_pairs[idx]

        low_samples = [self._load_sample(self.low_df.iloc[i]) for i in low_group]
        high_samples = [self._load_sample(self.high_df.iloc[i]) for i in high_group]

        # Filtra None (amostras corrompidas)
        low_samples = [s for s in low_samples if s is not None]
        high_samples = [s for s in high_samples if s is not None]

        if not low_samples or not high_samples:
            # Se faltar amostras, tenta outro grupo
            for attempt in range(5):
                alt_idx = np.random.randint(len(self.group_pairs))
                alt_low_group, alt_high_group = self.group_pairs[alt_idx]
                low_samples = [self._load_sample(self.low_df.iloc[i]) for i in alt_low_group]
                high_samples = [self._load_sample(self.high_df.iloc[i]) for i in alt_high_group]
                low_samples = [s for s in low_samples if s is not None]
                high_samples = [s for s in high_samples if s is not None]
                if low_samples and high_samples:
                    break

        return low_samples, high_samples

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
    groups=False,
    column_groups_id=None,
    score_col="arousal_score",
    binary_label=False,
    threshold=0.5,
    target_shape=TARGET_SHAPE,
    is_grayscale=False,
    video_transform=None,
    mel_transform=None,
    dtype=torch.float32,
    pin_memory=False,
    epoch_frac=1.0,
):
    """
    epoch_frac: fração (0 < epoch_frac <= 1) dos grupos/pares de grupos usados
                em cada época. Só tem efeito quando groups=True (é onde o
                sampler controla a iteração). Um subconjunto novo é sorteado
                a cada época. epoch_frac=1.0 (padrão) usa todos os dados.
    """
    if groups and column_groups_id is None:
        raise ValueError("column_groups_id deve ser informado quando groups=True.")

    if not groups and epoch_frac < 1.0:
        raise ValueError(
            "epoch_frac < 1.0 só tem efeito com groups=True "
            "(o sampler de grupos é quem controla a subamostragem por época)."
        )

    dataset = MultiModalDataset(
        csv_path=csv_path,
        groups=groups,
        column_groups_id=column_groups_id,
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

    sampler = None

    if groups:
        if pair: # grupos pareados para o margin ranking
            sampler = PairGroupSampler(dataset.group_pairs, shuffle=shuffle, frac=epoch_frac,)
        else: # os grupos não serão usados no margin ranking
            sampler = GroupSampler(dataset.groups, shuffle=shuffle, frac=epoch_frac,)
        # sampler e shuffle são mutuamente exclusivos
        shuffle = False # o sampler já faz o shuffle entre os grupos a cada época, então o do DataLoader do torch é desativado

    collate_fn = (multimodal_pair_collate_fn if pair else multimodal_collate_fn)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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