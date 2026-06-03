from audio_mel_spectogram import AudioPreprocessor

from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.pyplot as plt
from pathlib import Path
import librosa.display
import numpy as np
import random
import cv2
import os

from pathlib import Path
class VideoAudioGetSequences:
    '''mean
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

    def get_audio_array(self, audio, start, end):
        clip = audio.subclipped(start, end)
        audio_array = clip.to_soundarray(fps=self.sample_rate)

        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        return audio_array
    

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


    def save_mel_spectrogram(self, audio_array, output_path):
        mel = self.audio_processor.extrair_mel_spectograma(audio_array)
        np.save(str(output_path), mel.astype(np.float32))


    def is_silent(self, audio, threshold=1e-4):
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < threshold


    def preprocess(
        self,
        vid_path,
        highlights,
        video_format='mp4',
        with_audio=False,
        grayscale=False,
        fps=None,
        save_audio=False
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
            has_audio = audio is not None # evitar erros

            # Salva clipes dos segmentos highlight
            for idx, (start, end) in enumerate(highlights):
                output_path_video = highlight_video_dir / f"{video_name}_highlight_{idx}.{video_format}"
                self.save_video_clip(video, start, end, output_path_video, fps=fps, grayscale=grayscale, with_audio=with_audio)

                # Só processa áudio/mel se a trilha sonora existir no vídeo original
                if has_audio:
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

                # Proteção aplicada também na geração dos não-highlights
                if has_audio:
                    output_path_mel = no_highlight_mel_dir / f"{video_name}_no_highlight_{count}.npy"
                    self.save_mel_spectrogram(audio, start, end, output_path_mel)

                    if save_audio:
                        output_path_audio = no_highlight_audio_dir / f"{video_name}_no_highlight_{count}.wav"
                        self.save_audio_clip(audio, start, end, output_path_audio)
                count += 1

            if count < qtd_negativos:
                print(f"Apenas {count} segmentos negativos puderam ser gerados.")


    def save_segments(
        self,
        vid_path,
        intervals: list,
        output_dir,
        split: str, # train, val ou test
        prefix="clip",
        video_format='mp4',
        with_audio=False,
        grayscale=False,
        fps=None,
        save_audio=False,
        half = None # se refere ao tempo 1 ou 2 do jogo
    ):
        '''
        Salva clips de vídeo, áudio (opcional) e mel spectrograma a partir de intervalos definidos por pré definidos por VideoSlicer

        Parâmetros:
            vid_path      : caminho do vídeo de entrada.
            intervals    : lista de tuplas (inicio, fim) em HH:MM:SS ou segundos.
            video_format  : extensão do vídeo de saída (default: 'mp4').
            with_audio    : salva o vídeo com áudio (default: False).
            grayscale     : salva o vídeo em escala de cinza (default: False).
            fps           : frames por segundo; se None, usa o fps original.
            save_audio    : salva clipes de áudio em .wav (default: False).

            output dir : pasta que serão criadas as pastas de treino e teste
            split: (train, val ou test) vai salvar na pasta do train, val ou test
            half: (1 ou 2) passar para definir 1° ou 2° tempo da partida
        '''
        output_dir = Path(output_dir)

        # criando pastas para cada um dos conjuntos {train, val, test}:
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        test_dir = output_dir / 'test'

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # se o fps não foi passado, vai usar o do vídeo original
        if fps is None:
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        splits_path = {
            'train': output_dir / 'train',
            'val': output_dir / 'val',
            'test': output_dir / 'test'
        }

        for sp in splits_path:
            os.makedirs(sp, exist_ok=True)

        if split in splits_path:
            partida = os.path.basename(os.path.dirname(vid_path))
            output_path = splits_path[split] / partida / f"half_{half}"
            
            # Criamos as pastas específicas desta partida dinamicamente aqui:
            os.makedirs(output_path / "video", exist_ok=True)
            os.makedirs(output_path / "mel_spectograma", exist_ok=True)
            if save_audio:
                os.makedirs(output_path / "audio", exist_ok=True)
        else:
            raise ValueError(f"Split '{split}' inválido. Escolha entre 'train', 'val' ou 'test'.")

        with VideoFileClip(vid_path) as video:
            audio = video.audio

            for idx, (start, end) in enumerate(intervals):
                start = self.time_to_seconds(start)
                end = self.time_to_seconds(end)
                
                # vídeo
                output_video = (output_path / "video" /f"{prefix}_{idx}.{video_format}")
                self.save_video_clip(video, start, end, output_video, fps=fps, grayscale=grayscale, with_audio=with_audio)
                
                audio_array = self.get_audio_array(audio, start, end)

                if self.is_silent(audio_array): # não vai processar e salvar mel spectograma e audio
                    continue

                # mel spectograma
                output_mel = (output_path / "mel_spectograma" / f"{prefix}_{idx}.npy")
                self.save_mel_spectrogram(audio_array, output_mel)
                # audio (opcional)
                if save_audio:
                    output_audio = (output_path / "audio" / f"{prefix}_{idx}.wav")
                    self.save_audio_clip(audio, start, end, output_audio)


    def plot_mel_spectrogram(self, mel, title="Mel Spectrograma"):
        '''
        Plota um mel spectrograma salvo em memória.
        - mel (np.ndarray): Matriz do mel spectrograma.
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel, sr=self.sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def plot_saved_mel_spectrogram(self, mel_path):
        #Carrega e plota um mel spectrograma salvo em .npy.
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

"""
# ==============================================================================
# Execução: Processando ambos os tempos (Half 1 e Half 2)
# ==============================================================================
from video_slicer import VideoSlicer

# Caminho base para a pasta do jogo
game_dir = "/home/leticia/football/gaussian_football/data/raw/2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal"
title_dir = 'val'

# Loop para processar o tempo 1 e o tempo 2
for half in [1, 2]:
    vid_path = os.path.join(game_dir, f"{half}_224p.mkv")
    
    # Verifica se o arquivo de vídeo realmente existe antes de iniciar
    if not os.path.exists(vid_path):
        print(f"\n[AVISO] Arquivo não encontrado: {vid_path}. Pulando para o próximo.")
        continue
        
    print(f"\n" + "="*50)
    print(f"Iniciando processamento: TEMPO {half}")
    print(f"="*50)

    # Configura o fatiador
    slicer = VideoSlicer(n_slices=20)
    slices_list = slicer.get_intervals(video_path=vid_path)

    processor = VideoAudioGetSequences(clip_size=10)
    output_dir = 'data/processed' 

    processor.save_segments(
        vid_path=vid_path,
        intervals=slices_list,
        output_dir=output_dir,
        split = title_dir,
        grayscale=True,
        half=half,
        fps=15
    )
"""
'''
Resultado desse teste:
data/processed
    - nome da partida
        - pastas para half 1 ou 2
            - train, val e test
                - mel_spectograma e video
'''
print("\n[SUCESSO] Processamento concluído para ambas as partes!")