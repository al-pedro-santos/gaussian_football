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
    # converte "half - mm:ss" para (half, segundos)
    half, brute_time = game_time.split(" - ")
    min, sec = brute_time.split(":")
    return int(half), int(min) * 60 + int(sec)


def extract_goals(labels_path):
    # extrai os timestamps dos gols do Labels-v2.json
    with open(labels_path) as f:
        data = json.load(f)

    goals = []
    for event in data["annotations"]:
        if event["label"] == "Goal":
            goals.append(parse_game_time(event["gameTime"]))

    return goals


def load_games_index(index_path):
    # carrega um csv de índice de jogos e devolve apenas as linhas marcadas como válidas.
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["valid"] == "True"]


def goals_to_intervals(
    goals, duration, current_half, window_before=15, window_after=15
):
    # gera pequenos intervalos de tempo em torno de cada gol no tempo determinado, respeitando os limites do tempo total da metade.
    intervals = []

    for half, seconds in goals:
        if half != current_half:
            continue

        start = max(0, seconds - window_before)
        end = min(seconds + window_after, duration)

        intervals.append((half, start, end))
    return intervals


def build_arousal_timeline(video_path, intervals, half, fps):
    with VideoFileClip(video_path) as video:
        duration = math.ceil(video.duration)  # retorna segundos em float

    half_intervals = []  # intervalos de um dos tempos da partida

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
                        f"\n  ✗ Erro no clip {idx} ({start:.1f}s-{end:.1f}s): {e}, pulando..."
                    )
                    continue


def build_labels(
    game_dir, clips_dir, intervals_per_half, window_before=15, window_after=15, fps=25
):
    labels_path = os.path.join(game_dir, "Labels-v2.json")
    goals = extract_goals(labels_path)

    rows = []

    for half in [1, 2]:
        video_path = os.path.join(game_dir, f"{half}_224p.mkv")
        # cria intervalos usando a duração desse half
        with VideoFileClip(video_path) as video:
            duration = math.floor(video.duration)

        intervals = goals_to_intervals(
            goals,
            duration,
            half,
            window_before=window_before,
            window_after=window_after,
        )  # constrói os intervalos

        timeline = build_arousal_timeline(
            video_path, intervals, half, fps
        )  # usa os intervalos para construir a timeline

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
            print(f"  ✗ Erro ao ler vídeo, pulando: {e}")
            games_pulados += 1
            continue

        video_dir_half1 = os.path.join(clips_dir, "half_1", "video")
        clips_existem = (
            os.path.isdir(video_dir_half1) and len(os.listdir(video_dir_half1)) > 0
        )

        if clips_existem:
            print(f"  ✓ Clips já existem, pulando geração...")
        else:
            print(f"  → Gerando clips...")
            generate_clips(game_dir, clips_dir, intervals_per_half, fps=args.fps)

        if os.path.exists(args.output):
            df_existing = pd.read_csv(args.output)
            if game["game_id"] in df_existing["game_id"].values:
                print(f"  ✓ Labels já existem, pulando...")
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
        print(f"  ✓ {len(rows)} clips gerados | acumulado: {len(df)} clips")
