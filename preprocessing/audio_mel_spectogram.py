import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import os
import random


class AudioPreprocessor:
    def __init__(self, n_mels=128, n_fft=2048, hop_length=512, sample_rate=22050):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def extrair_mel_spectograma(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def plotar_mel_spectograma(self, audio):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            self.extrair_mel_spectograma(audio),
            sr=self.sample_rate,
            x_axis="time",
            y_axis="mel",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrograma")
        plt.tight_layout()
        plt.show()
