import cv2
import numpy as np
import matplotlib.pyplot as plt


class VideoScorerPreprocessor:
    """
    Constrói uma timeline temporal de scores para highlights em vídeos
    utilizando gaussianas.

    Cada highlight é modelado como uma gaussiana centrada no ponto médio
    do intervalo anotado. A variância é calculada de forma que o score
    nas bordas do intervalo seja igual ao threshold definido.

    Assim:
        - frames dentro do intervalo possuem score >= threshold;
        - frames fora possuem score < threshold.

    Parameters
    ----------
    video_path : str
        Caminho do vídeo.

    highlights : list of tuple
        Lista de highlights no formato:
            [(inicio, fim), ...]

    threshold : float, default=0.5
        Threshold mínimo para considerar um frame como highlight.
    """

    def __init__(self, video_path, highlights, threshold=0.5):
        self.video_path = video_path
        self.highlights = highlights # lista de highlights

        cap = cv2.VideoCapture(video_path)

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        self.timeline = np.zeros(self.frame_count) # timeline

        self.threshold = threshold

        self.gaussian_timeline = self.gaussian_timeline() # construindo a timeline de gaussianas

    def time_to_seconds(self, t):
        '''
        Converte HH:MM:SS para segundos.
        '''
        if isinstance(t, (int, float)):
            return float(t)
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def seconds_to_frame(self, s):
        return int(self.fps * s)

    def gaussian(self, x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def gaussian_timeline(self):
        x = np.arange(self.frame_count)

        for start, end in self.highlights:
            # converte para segundos:
            start_s = self.time_to_seconds(start)
            end_s = self.time_to_seconds(end)

            # converte para frame:
            start_f = self.seconds_to_frame(start_s)
            end_f = self.seconds_to_frame(end_s)

            mu = (start_f + end_f) / 2 # centro do highlight
            d = (end_f - start_f) / 2 # distância do centro até um dos limites do intervalo
            sigma = d / (np.sqrt(-2 * np.log(self.threshold))) # sigma determinado tal que o scorer do highlight está acima do threshold
            gaussian = self.gaussian(x, mu, sigma) # cria gaussiana
            self.timeline = np.maximum(self.timeline, gaussian) # evita soma de caudas e a soma de não highlight ultrapassar o threshold

        return self.timeline

    def plot(self):
        plt.figure(figsize=(15, 4))
        plt.plot(self.timeline)
        plt.axhline(y=self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold}')
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Highlight Score")

        plt.title("Temporal Highlight Scoring")
        plt.ylim(0, 1.05)
        plt.show()

# exemplo:
vid_path = '/home/leticia/football/teste_labeler.mp4'
highlights = [('00:00:15', '00:00:30'), ('00:00:33', '00:00:45'), ('00:02:20', '00:02:40')]
scorer = VideoScorerPreprocessor(video_path=vid_path, highlights=highlights, threshold=0.5)
scorer.plot()
