import math
from pathlib import Path
from moviepy import VideoFileClip


class VideoSlicer:
    '''
    Fatiador de vídeos baseado em intervalos de tempo.

    Parâmetros:
    - n_slices: quantidade de clips que o vídeo será cortado.
    '''

    def __init__(self, n_slices: int = 90):
        self.n_slices = n_slices

    def get_intervals(self, video_path: str, slice_length = None, n_slices = None):
        '''
        Recebe o caminho de um vídeo e retorna uma lista com (start, end) de cada clip sem sobreposição de intervalos

        - slice_lenght (Opcional): duração da janela (em segundos)
        - n_slices (Opcional): quantidade de clips que  o vídeo deve ser fateado

        ! apenas um dos dois deve ser passado como argumento: 'slice_length' ou 'n_slices'. !
        '''

        if slice_length is not None and n_slices is not None:
            raise ValueError(
                "Forneça apenas um dos argumentos: 'slice_length' ou 'n_slices'."
            )

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        with VideoFileClip(str(path)) as video:
            total = video.duration

        if slice_length is None:
            n = n_slices if n_slices is not None else self.n_slices
            length = total / n
        else:
            if slice_length > 0:
                length = slice_length
                n = math.ceil(total / length)

        intervals = []
        for i in range(n):
            start = i * length
            end = min(start + length, total)
            intervals.append((round(start, 6), round(end, 6)))

        return intervals
