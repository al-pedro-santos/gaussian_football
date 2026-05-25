from audio_mel_spectogram import AudioPreprocessor

from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display
import numpy as np
import random
import cv2
import os


class VideoAudioGetSequences:
    '''
    Gera clipes de vídeo, áudio e mel spectrogramas para segmentos
    highlight e no_highlight, produzindo um dataset balanceado.
    '''
    def __init__(self, clip_size=10, sample_rate=22050):
        self.clip_size = clip_size
        self.sample_rate = sample_rate
        self.audio_processor = AudioPreprocessor(sample_rate=sample_rate)

    def time_to_seconds(self, t):
        # Aceita segundos (int/float) ou string HH:MM:SS
        if isinstance(t, (int, float)):
            return float(t)
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def has_overlap(self, start, end, intervals):
        # Retorna True se [start, end] sobrepõe algum intervalo da lista
        for h_start, h_end in intervals:
            if (self.time_to_seconds(start) < self.time_to_seconds(h_end)
                    and self.time_to_seconds(end) > self.time_to_seconds(h_start)):
                return True
        return False

    def save_audio_clip(self, audio, start, end, output_path):
        # Recorta e salva trecho de áudio em .wav
        clip = audio.subclipped(start, end)
        clip.write_audiofile(str(output_path), fps=self.sample_rate, logger=None)

    def save_video_clip(self, video, start, end, output_path, fps, grayscale=False, with_audio=False):
        # Recorta e salva clipe de vídeo; grayscale e áudio são opcionais
        clip = video.subclipped(start, end)
        ffmpeg_params = ["-vf", "format=gray"] if grayscale else None
        clip.write_videofile(
            str(output_path), codec='libx264', audio=with_audio,
            ffmpeg_params=ffmpeg_params, logger=None, fps=fps
        )

    def save_mel_spectrogram(self, audio, start, end, output_path):
        # Extrai o trecho de áudio e salva o mel spectrograma em .npy
        clip = audio.subclipped(start, end)
        audio_array = clip.to_soundarray(fps=self.sample_rate)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        mel = self.audio_processor.extrair_mel_spectograma(audio_array)
        np.save(str(output_path), mel.astype(np.float32))

    def preprocess(
        self,
        vid_path,
        highlights,
        video_format='mp4',
        with_audio=False,
        grayscale=False,
        fps=None,
        save_audio=False,
    ):
        '''
        Processa um vídeo gerando clipes balanceados de highlight e no_highlight.

        Parâmetros:
            vid_path      : caminho do vídeo de entrada.
            highlights    : lista de tuplas (inicio, fim) em HH:MM:SS ou segundos.
            video_format  : extensão do vídeo de saída (default: 'mp4').
            with_audio    : salva o vídeo com áudio (default: False).
            grayscale     : salva o vídeo em escala de cinza (default: False).
            fps           : frames por segundo; se None, usa o fps original.
            save_audio    : salva clipes de áudio em .wav (default: False).

        Saída:
            parent/
              highlight/   | no_highlight/
                video/
                mel_spectograma/
                audio/     (apenas se save_audio=True)
        '''

        if fps is None:
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        highlights = [
            (self.time_to_seconds(start), self.time_to_seconds(end))
            for start, end in highlights
        ]

        parent = Path(vid_path).parent.parent
        video_name = Path(vid_path).stem

        # Cria diretórios fixos de saída
        highlight_dir      = parent / "highlight"
        no_highlight_dir   = parent / "no_highlight"

        highlight_video_dir    = highlight_dir    / "video"
        highlight_mel_dir      = highlight_dir    / "mel_spectograma"
        no_highlight_video_dir = no_highlight_dir / "video"
        no_highlight_mel_dir   = no_highlight_dir / "mel_spectograma"

        os.makedirs(highlight_video_dir,    exist_ok=True)
        os.makedirs(highlight_mel_dir,      exist_ok=True)
        os.makedirs(no_highlight_video_dir, exist_ok=True)
        os.makedirs(no_highlight_mel_dir,   exist_ok=True)

        # Cria pastas de áudio somente se necessário
        if save_audio:
            highlight_audio_dir    = highlight_dir    / "audio"
            no_highlight_audio_dir = no_highlight_dir / "audio"
            os.makedirs(highlight_audio_dir,    exist_ok=True)
            os.makedirs(no_highlight_audio_dir, exist_ok=True)

        with VideoFileClip(vid_path) as video:
            audio    = video.audio
            duration = video.duration

            # Salva clipes dos segmentos highlight
            for idx, (start, end) in enumerate(highlights):
                output_path_video = highlight_video_dir / f"{video_name}_highlight_{idx}.{video_format}"
                self.save_video_clip(video, start, end, output_path_video, fps=fps, grayscale=grayscale, with_audio=with_audio)

                output_path_mel = highlight_mel_dir / f"{video_name}_highlight_{idx}.npy"
                self.save_mel_spectrogram(audio, start, end, output_path_mel)

                if save_audio:
                    output_path_audio = highlight_audio_dir / f"{video_name}_highlight_{idx}.wav"
                    self.save_audio_clip(audio, start, end, output_path_audio)

            # Gera segmentos no_highlight sem sobreposição, balanceando o dataset
            generated_intervals = []
            qtd_negativos = len(highlights)
            count, attempts = 0, 0

            while count < qtd_negativos and attempts < 1000:
                attempts += 1
                start = random.uniform(0, duration - self.clip_size)
                end   = start + self.clip_size

                if self.has_overlap(start, end, highlights):
                    continue
                if self.has_overlap(start, end, generated_intervals):
                    continue

                generated_intervals.append((start, end))

                output_path_video = no_highlight_video_dir / f"{video_name}_no_highlight_{count}.{video_format}"
                self.save_video_clip(video, start, end, output_path_video, fps=fps, grayscale=grayscale, with_audio=with_audio)

                output_path_mel = no_highlight_mel_dir / f"{video_name}_no_highlight_{count}.npy"
                self.save_mel_spectrogram(audio, start, end, output_path_mel)

                if save_audio:
                    output_path_audio = no_highlight_audio_dir / f"{video_name}_no_highlight_{count}.wav"
                    self.save_audio_clip(audio, start, end, output_path_audio)

                count += 1

            if count < qtd_negativos:
                print(f"Apenas {count} segmentos negativos puderam ser gerados.")


    def plot_mel_spectrogram(self, mel, title="Mel Spectrograma"):
        '''
        Plota um mel spectrograma salvo em memória.

        Parameters
        ----------
        mel : np.ndarray
            Matriz do mel spectrograma.
        
        title : str
            Título do gráfico.
        '''

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel, sr=self.sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_saved_mel_spectrogram(self, mel_path):
        '''
        Carrega e plota um mel spectrograma salvo em .npy.
        '''
        mel = np.load(mel_path)

        self.plot_mel_spectrogram(mel, title=Path(mel_path).stem)


"""
Dicas para reduzir custo computacional e uso de memória no preprocess:

    - fps: use um fps menor que o original (ex: 15) para reduzir o número
      de frames processados e o tamanho dos arquivos de vídeo gerados.

    - grayscale: ativar converte os frames para escala de cinza, reduzindo
      o volume de dados do vídeo em ~3x (elimina os canais RGB).

    - with_audio: manter False (default) evita a codificação de áudio
      junto ao vídeo, acelerando o write_videofile.

    - save_audio: manter False (default) evita salvar arquivos .wav
      separados, economizando I/O de disco e memória durante a extração.

    - clip_size: clipes mais curtos nos segmentos no_highlight reduzem
      o tempo de recorte e o tamanho dos arquivos gerados.

    - sample_rate: um sample_rate menor (ex: 16000 vs 22050) reduz o
      tamanho dos arrays de áudio e o custo do cálculo do mel spectrograma.
"""