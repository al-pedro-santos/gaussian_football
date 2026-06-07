import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing")
)

import csv
import math
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from video_scorer import VideoScorerPreprocessor
from video_slicer import VideoSlicer
from video_and_audio_get_sequences import VideoAudioGetSequences
from moviepy.video.io.VideoFileClip import VideoFileClip

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")


def parse_game_time(game_time):
    """Converte "half - mm:ss" para (half, seconds).

    Args:
        game_time (str): Momento do jogo.

    Returns:
        tuple[int, int]: Tupla (half, seconds), onde half é o tempo
        da partida (1 ou 2) e seconds é o instante convertido para
        segundos.
    """
    half, brute_time = game_time.split(" - ")
    min, sec = brute_time.split(":")
    return int(half), int(min) * 60 + int(sec)


def extract_goals(labels_path):
    """Extrai os timestamps dos gols de uma partida a partir de Labels-v2.json.

    Args:
        labels_path (str): Path do Labels-v2.json de uma partida.

    Returns:
        list[tuple[int, int]]: Lista de tuplas (half, seconds), uma
            para cada gol anotado na partida.
    """
    with open(labels_path) as f:
        data = json.load(f)

    goals = []
    for event in data["annotations"]:
        if event["label"] == "Goal":
            goals.append(parse_game_time(event["gameTime"]))

    return goals


def load_games_index(index_path):
    """Carrega o CSV de índice de jogos e retorna apenas as linhas válidas.

    Args:
        index_path (str): Path do CSV de índice (games_index.csv).

    Returns:
        list[dict]: Lista de dicionários, um por jogo válido (coluna
            'valid' == 'True'), com as colunas do índice como chaves.
    """
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["valid"] == "True"]


def goals_to_intervals(
    goals, duration, current_half, window_before=15, window_after=15
):
    """Gera intervalos de tempo ao redor de cada gol de um dado tempo da partida.

    Para cada gol no tempo (half) especificado, cria um intervalo
    [gol - window_before, gol + window_after], respeitando os limites
    [0, duration] da metade.

    Args:
        goals (list[tuple[int, int]]): Lista de tuplas (half, seconds)
            retornada por extract_goals.
        duration (int): Duração total do tempo atual em segundos.
        current_half (int): Tempo da partida (1 ou 2).
        window_before (int): Segundos antes do gol a incluir no intervalo.
            Default: 15.
        window_after (int): Segundos após o gol a incluir no intervalo.
            Default: 15.

    Returns:
        list[tuple[int, int, int]]: Lista de tuplas (half, start, end), uma
            para cada gol do tempo especificado, com start e end em segundos.
    """
    intervals = []

    for half, seconds in goals:
        if half != current_half:
            continue

        start = max(0, seconds - window_before)
        end = min(seconds + window_after, duration)

        intervals.append((half, start, end))
    return intervals


def build_arousal_timeline(video_path, intervals, half, fps):
    """Constrói a timeline de arousal de um tempo da partida.

    Filtra os intervalos do tempo especificado, recorta-os aos limites do
    vídeo e os repassa ao VideoScorerPreprocessor, que modela cada intervalo
    como uma gaussiana.

    Args:
        video_path (str): Path do vídeo do tempo da partida.
        intervals (list[tuple[int, int, int]]): Tuplas (half, start, end).
        half (int): Tempo da partida a processar (1 ou 2).
        fps (int): Frames por segundo, para converter segundos em frames.

    Returns:
        np.ndarray: Vetor 1D com o arousal score de cada frame no intervalo [0, 1].
    """
    with VideoFileClip(video_path) as video:
        duration = math.ceil(video.duration)

    half_intervals = []

    for h, start, end in intervals:
        if h != half:
            continue

        start = max(0, start)
        end = min(end, duration)

        if end > start:
            half_intervals.append((start, end))

    scorer = VideoScorerPreprocessor(video_path, half_intervals, fps=fps)
    return scorer.gaussian_timeline


def generate_clips(game_dir, clips_dir, intervals_per_half, fps=25, grayscale=True):
    """Gera os clipes de vídeo e mel espectrogramas de uma partida.

    Para cada tempo da partida, percorre os intervalos fornecidos e
    salva um clipe de vídeo por intervalo. Quando o áudio existe e não
    é silencioso, salva também o mel espectrograma. Erros em clipes
    individuais são capturados e o processamento continua.

    Os arquivos são organizados em:
        clips_dir/half_{n}/video/clip_{idx}.mp4
        clips_dir/half_{n}/mel_spectograma/clip_{idx}.npy

    Args:
        game_dir (str): Diretório da partida, contendo os arquivos
            {half}_224p.mkv.
        clips_dir (str): Diretório de saída onde os clipes serão salvos.
        intervals_per_half (dict[int, list[tuple[float, float]]]): Intervalos
            (start, end) por tempo gerados pelo VideoSlicer.
        fps (int): Frames por segundo dos clipes de saída. Default: 25 (SoccerNet).
        grayscale (bool): Se True, salva os clipes em escala de cinza.
            Default: True.

    Returns:
        None: Os clipes são salvos diretamente em disco.
    """
    processor = VideoAudioGetSequences()

    for half in [1, 2]:
        video_path = os.path.join(game_dir, f"{half}_224p.mkv")

        if not os.path.exists(video_path):
            print(f"Vídeo não encontrado: {video_path}, pulando...")
            continue

        output_dir = os.path.join(clips_dir, f"half_{half}")
        os.makedirs(os.path.join(output_dir, "video"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mel_spectograma"), exist_ok=True)

        with VideoFileClip(video_path) as video:
            audio = video.audio

            for idx, (start, end) in enumerate(
                tqdm(intervals_per_half[half], desc=f"Gerando clips Half {half}")
            ):
                try:
                    output_video = os.path.join(output_dir, "video", f"clip_{idx}.mp4")
                    processor.save_video_clip(
                        video, start, end, output_video, fps=fps, grayscale=grayscale
                    )

                    if audio is not None:
                        audio_array = processor.get_audio_array(audio, start, end)
                        if not processor.is_silent(audio_array):
                            output_mel = os.path.join(
                                output_dir, "mel_spectograma", f"clip_{idx}.npy"
                            )
                            processor.save_mel_spectrogram(audio_array, output_mel)
                except Exception as e:
                    print(
                        f"\n  Erro no clip {idx} ({start:.1f}s-{end:.1f}s): {e}, pulando..."
                    )
                    continue


def build_labels(
    game_dir, clips_dir, intervals_per_half, window_before=15, window_after=15, fps=25
):
    """Calcula o arousal score de cada clipe de uma partida.

    Para cada tempo, constrói a timeline de arousal via gaussianas e atribui
    a cada clipe o valor máximo (np.max) da timeline em seu intervalo.

    Args:
        game_dir (str): Diretório da partida ({half}_224p.mkv, Labels-v2.json).
        clips_dir (str): Diretório onde os clipes devem ser gerados.
        intervals_per_half (dict[int, list[tuple[float, float]]]): Intervalos
            (start, end) por tempo.
        window_before (int): Segundos antes do gol na janela. Default: 15.
        window_after (int): Segundos após o gol na janela. Default: 15.
        fps (int): Frames por segundo. Default: 25.

    Returns:
        list[dict]: Um dict por clipe com 'clip_path', 'mel_path' (ou None)
            e 'arousal_score'.
    """
    labels_path = os.path.join(game_dir, "Labels-v2.json")
    goals = extract_goals(labels_path)

    rows = []

    for half in [1, 2]:
        video_path = os.path.join(game_dir, f"{half}_224p.mkv")

        with VideoFileClip(video_path) as video:
            duration = math.floor(video.duration)

        intervals = goals_to_intervals(
            goals,
            duration,
            half,
            window_before=window_before,
            window_after=window_after,
        )

        timeline = build_arousal_timeline(video_path, intervals, half, fps)

        half_clips_dir = os.path.join(clips_dir, f"half_{half}")
        video_dir = os.path.join(half_clips_dir, "video")
        mel_dir = os.path.join(half_clips_dir, "mel_spectograma")

        clip_files = sorted(
            os.listdir(video_dir), key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        for idx, (start, end) in enumerate(
            tqdm(intervals_per_half[half], desc=f"Half {half}")
        ):
            clip_file = clip_files[idx]

            clip_path = os.path.join(video_dir, clip_file)
            mel_path = os.path.join(mel_dir, clip_file.replace(".mp4", ".npy"))

            if not os.path.exists(mel_path):
                mel_path = None

            start_frame = int(start * fps)
            end_frame = int(end * fps)

            arousal_score = float(np.max(timeline[start_frame:end_frame]))

            rows.append(
                {
                    "clip_path": clip_path,
                    "mel_path": mel_path,
                    "arousal_score": arousal_score,
                }
            )

    return rows


def main():
    """Executa o pipeline de geração de clipes e labels para todos os jogos.

    Lê o índice de jogos válidos, fatia cada partida via VideoSlicer, gera os
    clipes e calcula os arousal scores, colocando tudo em um CSV.  Pula
    clipes e labels já existentes de forma independente, evitando reprocessamento.

    Argumentos de linha de comando:
        --index_path:    Caminho do games_index.csv de entrada.
        --processed_dir: Diretório raiz onde os clipes são salvos.
        --fps:           Frames por segundo dos clipes. Default: 25.
        --n_slices:      Clipes por tempo da partida. Default: 330.
        --window_before: Segundos antes do gol na janela. Default: 15.
        --window_after:  Segundos após o gol na janela. Default: 15.
        --output:        Caminho do labels_all.csv de saída.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="data/labels/games_index.csv")
    parser.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--n_slices", type=int, default=330)
    parser.add_argument("--window_before", type=int, default=15)
    parser.add_argument("--window_after", type=int, default=15)
    parser.add_argument("--output", default="data/labels/labels_all.csv")
    args = parser.parse_args()

    slicer = VideoSlicer(n_slices=args.n_slices)
    games = load_games_index(args.index_path)

    total_games = len(games)
    games_processados = 0
    games_pulados = 0
    total_clips = 0

    print(f"Total de jogos no index: {total_games}")

    all_rows = []
    for i, game in enumerate(games, 1):
        game_dir = os.path.dirname(game["1_224p.mkv"])
        clips_dir = os.path.join(
            args.processed_dir,
            game["league"],
            game["season"],
            os.path.basename(game_dir),
        )

        print(f"\n[{i}/{total_games}] {game['game_id']}")

        intervals_per_half = {}
        try:
            for half in [1, 2]:
                video_path = os.path.join(game_dir, f"{half}_224p.mkv")
                intervals_per_half[half] = slicer.get_intervals(video_path)
        except OSError as e:
            print(f" Erro ao ler vídeo, pulando: {e}")
            games_pulados += 1
            continue

        video_dir_half1 = os.path.join(clips_dir, "half_1", "video")
        clips_existem = (
            os.path.isdir(video_dir_half1) and len(os.listdir(video_dir_half1)) > 0
        )

        if clips_existem:
            print(f" Clips já existem, pulando geração...")
        else:
            print(f" Vai gerar clips agora para {game['game_id']}")
            generate_clips(game_dir, clips_dir, intervals_per_half, fps=args.fps)

        if os.path.exists(args.output):
            df_existing = pd.read_csv(args.output)
            if game["game_id"] in df_existing["game_id"].values:
                print(f" Labels já existem, pulando...")
                all_rows.extend(
                    df_existing[df_existing["game_id"] == game["game_id"]].to_dict(
                        "records"
                    )
                )
                games_processados += 1
                continue

        rows = build_labels(
            game_dir,
            clips_dir,
            intervals_per_half,
            window_before=args.window_before,
            window_after=args.window_after,
            fps=args.fps,
        )

        for row in rows:
            row["game_id"] = game["game_id"]
            row["season"] = game["season"]
            row["split"] = game["split"]

        all_rows.extend(rows)
        games_processados += 1
        total_clips += len(rows)

        df = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"{len(rows)} clips gerados | acumulado: {len(df)} clips")

    print(f"\n{'='*50}")
    print(f"Concluído.")
    print(f"  Jogos processados: {games_processados}/{total_games}")
    print(f"  Jogos pulados:     {games_pulados}")
    print(f"  Clips gerados nesta execução: {total_clips}")
    print(f"  Total de linhas no CSV: {len(all_rows)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
