import cv2
import numpy as np
import matplotlib.pyplot as plt


class VideoScorerPreprocessor:
    def __init__(self, video_path, highlights, k=5):
        self.video_path = video_path
        self.highlights = highlights # lista de highlights

        cap = cv2.VideoCapture(video_path)

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        self.timeline = np.zeros(self.frame_count) # timeline

        # hiperparâmetro em segundos
        self.k = k

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
            # converte para segundos
            start_s = self.time_to_seconds(start)
            end_s = self.time_to_seconds(end)

            # converte para frame
            start_f = self.seconds_to_frame(start_s)
            end_f = self.seconds_to_frame(end_s)

            # centro do highlight
            mu = (start_f + end_f) / 2

            # duração do highlight
            duration = end_f - start_f

            # sigma proporcional ao highlight
            sigma = duration / self.k

            # evita sigma muito pequeno
            sigma = max(sigma, 1)

            # cria gaussiana
            gaussian = self.gaussian(x, mu, sigma)

            # evita soma de caudas e a soma de não highlight ultrapassar o threshold
            self.timeline = np.maximum(self.timeline, gaussian)

        return self.timeline

    def plot(self):
        plt.figure(figsize=(15, 4))
        plt.plot(self.timeline)

        plt.xlabel("Frame")
        plt.ylabel("Highlight Score")

        plt.title("Temporal Highlight Scoring")
        plt.ylim(0, 1)
        plt.show()

# exemplo:
'''
vid_path = '/home/al.leticia.ferreira/football/test/teste_labeler.mp4'
highlights = [('00:00:15', '00:00:30'), ('00:02:20', '00:02:40')]
scorer = VideoScorerPreprocessor(video_path=vid_path, highlights=highlights)
scorer.plot()
'''
