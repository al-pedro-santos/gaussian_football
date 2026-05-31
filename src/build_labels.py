import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))

import json
import csv
import argparse
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from video_scorer import VideoScorerPreprocessor

def parse_game_time(game_time):
    # pegando o tempo e o instante em segundos do gol
    half, brute_time = game_time.split(" - ")
    min, sec = brute_time.split(":")
    return int(half), int(min) * 60 + int(sec)

def extract_goals(labels_path):
    # extraindo os gols de uma partida
    with open(labels_path) as f:
        data = json.load(f)
    
    goals = []
    for event in data["annotations"]:
        if event["label"] == "Goal":
            goals.append(parse_game_time(event["gameTime"]))

    return goals

def goals_to_intervals(goals, window_before=15, window_after=15):
    # passando cada instante de gol para um intervalo
    return [(half, max(0, seconds - window_before), seconds + window_after) for (half, seconds) in goals]

def build_arousal_timeline(video_path, intervals, half):

    """
    video_path: caminho do .mp4 do tempo correspondente
    intervals: lista de tuplas (half, start, end)
    half: 1 ou 2, qual tempo processar
    """

    # como cada tempo é um vídeo separado, lida com um tempo de cada vez
    half_intervals = [(start, end) for h, start, end in intervals if h == half]
    scorer = VideoScorerPreprocessor(video_path, half_intervals)
    return scorer.gaussian_timeline

def build_labels(game_dir, clips_dir, intervals_per_half, fps=25):
    labels_path = os.path.join(game_dir, "Labels-v2.json")
    goals = extract_goals(labels_path)
    intervals = goals_to_intervals(goals)

    rows = []
    for half in [1, 2]:
        video_path = os.path.join(game_dir, f"{half}_224p.mkv")
        timeline = build_arousal_timeline(video_path, intervals, half)

        half_clips_dir = os.path.join(clips_dir, f"half_{half}")
        video_dir = os.path.join(half_clips_dir, "video")
        mel_dir   = os.path.join(half_clips_dir, "mel_spectograma")

        clip_files = sorted(os.listdir(video_dir))

        for idx, (start, end) in enumerate(intervals_per_half[half]):
            clip_file = clip_files[idx]
            clip_path = os.path.join(video_dir, clip_file)
            mel_path  = os.path.join(mel_dir, clip_file.replace(".mp4", ".npy"))

            start_frame = int(start * fps)
            end_frame   = int(end * fps)
            arousal_score = float(np.mean(timeline[start_frame:end_frame]))

            rows.append({
                "clip_path": clip_path,
                "mel_path": mel_path,
                "arousal_score": arousal_score,
            })

    return rows