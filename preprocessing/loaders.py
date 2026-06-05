from torch.utils.data import DataLoader

from datasets_mel_video import MelSpectrogramDataset, VideoClipDataset, video_collate_fn


def build_mel_dataloader(
    csv_path,
    split,
    batch_size,
    shuffle,
    num_workers,
    score_col="arousal_score",
    binary_label=False,
    threshold=0.5,
    target_shape=(128, 256),
    transform=None,
    dtype=None,
    pin_memory=False,
):
    """
    Cria um DataLoader para mel spectrogramas.

    Args:
        csv_path (str):
            Caminho para o arquivo labels_all.csv.

        split (str):
            Conjunto a ser utilizado.
            Opções: {"train", "val", "test"}.

        batch_size (int):
            Quantidade de amostras por batch.

        shuffle (bool):
            Embaralha os dados a cada época.

        num_workers (int):
            Número de processos utilizados para carregar os dados.

        score_col (str):
            Nome da coluna utilizada como label.

        binary_label (bool):
            Se True, converte o score contínuo em label binária.

        threshold (float):
            Limiar utilizado para binarização.

        target_shape (tuple[int, int]):
            Tamanho final do mel spectrograma
            (n_mels, n_frames).

        transform:
            Transformações aplicadas ao mel spectrograma.

        dtype:
            Tipo do tensor retornado.

        pin_memory (bool):
            Ativa pin_memory no DataLoader.

    Returns:
        DataLoader
    """

    dataset = MelSpectrogramDataset(
        csv_path=csv_path,
        split=split,
        score_col=score_col,
        binary_label=binary_label,
        threshold=threshold,
        target_shape=target_shape,
        transform=transform,
        dtype=dtype,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_video_dataloader(
    csv_path,
    split,
    batch_size,
    shuffle,
    num_workers,
    score_col="arousal_score",
    binary_label=False,
    threshold=0.5,
    is_grayscale=False,
    transform=None,
    dtype=None,
    pin_memory=False,
):
    """
    Cria um DataLoader para clips de vídeo.

    Args:
        csv_path (str):
            Caminho para o arquivo labels_all.csv.

        split (str):
            Conjunto a ser utilizado.
            Opções: {"train", "val", "test"}.

        batch_size (int):
            Quantidade de amostras por batch.

        shuffle (bool):
            Embaralha os dados a cada época.

        num_workers (int):
            Número de processos utilizados para carregar os dados.

        score_col (str):
            Nome da coluna utilizada como label.

        binary_label (bool):
            Se True, converte o score contínuo em label binária.

        threshold (float):
            Limiar utilizado para binarização.

        is_grayscale (bool):
            Indica se os vídeos armazenados foram
            gerados em escala de cinza.

        transform:
            Transformações aplicadas frame a frame.

        dtype:
            Tipo do tensor retornado.

        pin_memory (bool):
            Ativa pin_memory no DataLoader.

    Returns:
        DataLoader
    """

    dataset = VideoClipDataset(
        csv_path=csv_path,
        split=split,
        score_col=score_col,
        binary_label=binary_label,
        threshold=threshold,
        is_grayscale=is_grayscale,
        transform=transform,
        dtype=dtype,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=video_collate_fn,
    )
