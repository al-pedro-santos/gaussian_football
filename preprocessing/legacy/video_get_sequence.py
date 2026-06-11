from pathlib import Path
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import random

class VideoGetSequences:
    '''
    - Geração de pastas com os momentos highlights e não highlights
    - A quantidade de amostras em cada classe serão iguais para gerar um dataset balanceado
    - A duração dos clipes não highlights deve ser dada
    '''
    def __init__ (self, clip_size=10):
        self.clip_size = clip_size #duração dos clipes que não são highlights, em segundos

    def time_to_seconds(self, t):
        '''
        Converte HH:MM:SS para segundos.
        '''
        if isinstance(t, (int, float)):
            return float(t)

        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def has_overlap(self, start, end, intervals):
        '''
        Verifica se o intervalo [start, end] tem sobreposição com algum dos intervalos em intervals.
        '''
        for h_start, h_end in intervals:
            if self.time_to_seconds(start) < self.time_to_seconds(h_end) and self.time_to_seconds(end) > self.time_to_seconds(h_start):
                return True
        return False
    

    def save_clip(self, video, start, end, output_path):
        '''
        Salva um clipe do vídeo entre os tempos start e end no caminho output_path.
        '''
        clip = video.subclipped(start, end)
        clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)


    def preprocess(self, vid_path, highlights):
        '''
        highlights: lista de tuplas [(inicio, fim), ...] representando os intervalos de tempo dos highlights.
        O formato do tempo na lista de tuplas pode ser HH:MM:SS.mmm, HH:MM:SS, ou conversão do tempo para segundos
        '''
        highlights = [(self.time_to_seconds(start), self.time_to_seconds(end)) for start, end in highlights]

        parent = Path(vid_path).parent.parent
        video_name = Path(vid_path).stem

        highlight_dir = parent / f"{Path(vid_path).parent.name}__highlight"
        no_highlight_dir = parent / f"{Path(vid_path).parent.name}__no_highlight"

        os.makedirs(highlight_dir, exist_ok=True)
        os.makedirs(no_highlight_dir, exist_ok=True)

        with VideoFileClip(vid_path) as video:
            duration = video.duration

            for idx, (start, end) in enumerate(highlights):

                output_name = f"{video_name}_highlight_{idx}.mp4"
                output_path = highlight_dir / output_name

                self.save_clip(video, start, end, output_path)

            # Gerar clipes não-highlight
            generated_intervals = []
            qtd_negativos = len(highlights) # Gerar dataset balanceado
            count = 0
            max_attempts = 1000
            attempts = 0

            while count < qtd_negativos and attempts < max_attempts:
                attempts += 1
                start = random.uniform(0, duration - self.clip_size)
                end = start + self.clip_size

                # Verifica overlap com highlights (os intervalos devem ser disjuntos)
                overlap_highlight = self.has_overlap(start, end, highlights)

                # Verifica overlap com negativos já criados
                overlap_negative = self.has_overlap(start, end, generated_intervals)

                if overlap_highlight or overlap_negative:
                    continue

                generated_intervals.append((start, end))

                output_name = f"{video_name}_no_highlight_{count}.mp4"
                output_path = no_highlight_dir / output_name

                self.save_clip(video, start, end, output_path)

                count += 1

            if count < qtd_negativos:
                print(f"Apenas {count} clips negativos puderam ser gerados.")