'''
A estrutura esperada do dataset, criada por build_labels.py é:
    labels.csv -> colunas: clip_path, mel_path, arousal_score

uso básico:
    mel_dataset = MelSpectogramaDataset("labels.csv")
    mel_loader = build_mel_dataloader(dataset, batch_size=32, shuffle=True)

Sobre pin_memory:
    Com pin_memory=True, o Torch coloca os batches em uma região da RAM chamada pinned memory,
    que permite transferências mais rápidas para a GPU.
'''

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class MelSpectogramDataset(Dataset):
    """
    Dataset para mel spectrogramas gerados por AudioPreprocessor ou
    VideoAudioGetSequences.

    Parâmetros:
    - csv_path:
        CSV com pelo menos as colunas `mel_path` e `arousal_score`.
    - score_col: str
        Nome da coluna usada como rótulo (default: "arousal_score").
    - binary: bool
        Se True, converte o rótulo contínuo em 0/1 usando `threshold`.
    - threshold: float
        Limiar para a conversão binária (default: 0.5).
    - transform:
        Transform aplicado ao tensor [1, n_mels, T].
    """
    def __init__(self, 
                 csv_path,
                 score_col = "arousal_score",
                 binary_label = False,
                 threshold = 0.5,
                 transform = None):
    
        self.df = pd.read_csv(csv_path)
        self.score_col = score_col
        self.binary_label = binary_label
        self.threshold = threshold
        self.transform = transform


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
        score_val = float(row[self.score_col])
        if self.binary_label: 
            score_val = float(score_val >= self.threshold) # classificação 0-1 para highlight
        score = torch.tensor(score_val, dtype=self.dtype)

        return mel_tensor, score
    
    @property
    def score(self):
        # retorna um numpy array com os rótulos (0-1 ou aurosal)
        vals = self.df[self.score_col].values.astype(float)
        if self.binary:
            vals = (vals >= self.threshold).astype(float)
        return vals
    

    def __repr__(self):
        return (f"MelSpectrogramDataset(n_samples={len(self)},\n score='{self.score_col}',\n binary_label={self.binary_label})")


def build_mel_dataloader(csv_path, batch_size, shuffle: bool, num_workers: int, pin_memory=False,
                         **dataset_kwargs): # passar os argumentos do MelSpectogramaDataset
    '''
    Cria Dataloader para treino/validação

    parâmetros extras em `dataset_kwargs` são repassados para
    MelSpectrogramDataset (score_col, binary, threshold, transform,...
    '''
    dataset = MelSpectogramDataset(csv_path, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

