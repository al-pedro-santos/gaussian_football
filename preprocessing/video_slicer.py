import math
from pathlib import Path
from moviepy import VideoFileClip


class VideoSlicer:
    """Fatiador de vídeos baseado em intervalos de tempo.

    Args:
        n_slices (int): Quantidade padrão de clipes a serem gerados.
            Pode ser sobrescrito pelo argumento n_slices do método
            get_intervals. Default: 330.
    """

    def __init__(self, n_slices: int = 330):
        self.n_slices = n_slices

    def get_intervals(self, video_path: str, slice_length=None, n_slices=None):
        """Retorna os intervalos de tempo para fatiamento de um vídeo.

        Apenas um dos argumentos opcionais deve ser fornecido: se ambos forem passados,
        o método levanta um erro. Se nenhum for fornecido, utiliza o valor de
        self.n_slices definido no construtor.

        Args:
            video_path (str): Caminho para o arquivo de vídeo.
            slice_length (float, optional): Duração de cada clipe em segundos.
            n_slices (int, optional): Quantidade de clipes a serem gerados.

        Returns:
            list[tuple[float, float]]: Lista de tuplas (start, end) representando
                os instantes inicial e final de cada intervalo em segundos.

        Raises:
            ValueError: Se ambos slice_length e n_slices forem fornecidos, ou
                se slice_length for menor ou igual a zero.
            FileNotFoundError: Se o arquivo de vídeo não for encontrado.
        """

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
            else:
                raise ValueError("slice_length deve ser maior que zero.")

        intervals = []
        for i in range(n):
            start = i * length
            end = min(start + length, total)
            intervals.append((round(start, 6), round(end, 6)))

        return intervals
