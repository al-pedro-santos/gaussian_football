import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))

import math
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessing.video_scorer import VideoScorerPreprocessor
from preprocessing.video_slicer import VideoSlicer
from moviepy.video.io.VideoFileClip import VideoFileClip

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

def goals_to_intervals(goals, duration, current_half, window_before=15, window_after=15):
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
        duration = math.ceil(video.duration) # retorna segundos em float

    half_intervals = [] # intervalos de um dos tempos da partida

    for h, start, end in intervals:
        if h != half:
            continue

        start = max(0, start)
        end = min(end, duration)

        if end > start:
            half_intervals.append((start, end))

    scorer = VideoScorerPreprocessor(video_path, half_intervals, fps=fps)
    return scorer.gaussian_timeline


def build_labels(game_dir, clips_dir, intervals_per_half, window_before=15, window_after=15, fps=25):
    labels_path = os.path.join(game_dir, "Labels-v2.json")
    goals = extract_goals(labels_path)

    rows = []

    for half in [1, 2]:
        video_path = os.path.join(game_dir, f"{half}_224p.mkv")
        # cria intervalos usando a duração desse half
        with VideoFileClip(video_path) as video:
            duration = math.floor(video.duration)

        intervals = goals_to_intervals(goals, duration, half, window_before=window_before, window_after=window_after) # constrói os intervalos

        timeline = build_arousal_timeline(video_path, intervals, half, fps) # usa os intervalos para construir a timeline

        half_clips_dir = os.path.join(clips_dir, f'half_{half}')
        video_dir = os.path.join(half_clips_dir, "video")
        mel_dir = os.path.join(half_clips_dir, "mel_spectograma")

        clip_files = sorted(
            os.listdir(video_dir),
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        for idx, (start, end) in enumerate(tqdm(intervals_per_half[half], desc=f"Half {half}")):
            clip_file = clip_files[idx]

            clip_path = os.path.join(video_dir, clip_file)
            mel_path = os.path.join(mel_dir, clip_file.replace(".mp4", ".npy"))

            if not os.path.exists(mel_path):  
                mel_path = None

            start_frame = int(start * fps)
            end_frame = int(end * fps)

            arousal_score = float(np.max(timeline[start_frame:end_frame]))

            rows.append({
                "clip_path": clip_path,
                "mel_path": mel_path,
                "arousal_score": arousal_score,
            })

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_dir",       required=True)
    parser.add_argument("--clips_dir",      required=True)
    parser.add_argument("--fps",            type=int, default=25)
    parser.add_argument("--n_slices",       type=int, default=330)
    parser.add_argument("--window_before",  type=int, default=15)  
    parser.add_argument("--window_after",   type=int, default=15)  
    parser.add_argument("--output",         default="labels.csv")
    args = parser.parse_args()

    slicer = VideoSlicer(n_slices=args.n_slices)
    intervals_per_half = {}
    for half in [1, 2]:
        video_path = os.path.join(args.game_dir, f"{half}_224p.mkv")
        intervals_per_half[half] = slicer.get_intervals(video_path)

    rows = build_labels(
        args.game_dir,
        args.clips_dir,
        intervals_per_half,
        window_before=args.window_before,
        window_after=args.window_after,
        fps=args.fps,
    )

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Salvo em {args.output} ({len(df)} clips)")

if __name__ == "__main__":
    main()