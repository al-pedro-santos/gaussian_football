
"""
preprocess.py
Phase 2 — Extrai clips de cada jogo e salva como tensores .npy.
 
Uso:
    python src/preprocess.py
    python src/preprocess.py --clip_duration 5 --n_frames 10
    python src/preprocess.py --games_index data/labels/games_index.csv
"""
 
from decord import VideoReader, cpu
import numpy as np
import os
import csv
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

# CONSTANTES


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GAMES_INDEX  = os.path.join(_PROJECT_ROOT, "data", "labels", "games_index.csv")
DEFAULT_CLIPS_INDEX  = os.path.join(_PROJECT_ROOT, "data", "labels", "clips_index.csv")
DEFAULT_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

# TRANSFORMAÇÃO DOS FRAMES

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# FUNÇÕES DE PROCESSAMENTO

def process_frame(frame):
    """Recebe frame RGB numpy (H, W, 3), devolve array (3, 224, 224) float32 normalizado."""
    return transform(frame).numpy() # (3, 224, 224) float32


def extract_frames(vr, start_frame, clip_frames, n_frames=10): # 5s a 2fps = 10 frames
    """
    vr          — VideoReader do decord
    start_frame — índice do primeiro frame do clip
    clip_frames — duração do clip em frames (fps * clip_duration)
    n_frames    — quantos frames extrair (default 10)
    """
    indices = np.linspace(start_frame, start_frame + clip_frames - 1, num=n_frames, dtype=int)
    frames = vr.get_batch(indices)  # (10, H, W, 3)
    return frames


def extract_clips(vr, clip_duration=5, n_frames=10):
    """
    Divide um vídeo em clips de duração fixa e extrai frames de cada um.

    vr            — VideoReader do decord
    clip_duration — duração de cada clip em segundos (default 5)
    n_frames      — frames a extrair por clip (default 10, equivale a 2fps)

    Retorna lista de tuplas (start_sec, clip) onde:
        start_sec — tempo de início do clip em segundos
        clip      — array numpy (n_frames, 3, 224, 224) float32 normalizado
    """
    fps = vr.get_avg_fps()
    clip_frames = int(fps * clip_duration)
    n_clips = len(vr) // clip_frames

    clips = []
    for i in range(n_clips):
        start_frame = clip_frames * i
        start_sec = start_frame / fps

        frames = extract_frames(vr, start_frame, clip_frames, n_frames)
        
        clip = []
        for f in range(n_frames):
            frame = process_frame(frames[f].asnumpy())
            clip.append(frame)
        
        clip = np.stack(clip)  # (10, 3, 224, 224)
        clips.append((start_sec, clip))

    return clips

def preprocess_game(row, processed_dir, clip_duration=5, n_frames=10):
    """
    Processa um jogo completo — abre os dois tempos, extrai clips,
    salva como .npy e retorna as linhas pro clips_index.csv.

    row           — linha do games_index.csv (pandas Series)
    processed_dir — pasta raiz de data/processed/
    clip_duration — duração de cada clip em segundos (default 5)
    n_frames      — frames por clip (default 10)

    Retorna lista de dicionários, um por clip.
    """
    video_paths = [row["1_224p.mkv"], row["2_224p.mkv"]]
    clips_data = []

    for half, video_path in enumerate(video_paths, start=1):
        vr = VideoReader(video_path, ctx=cpu(0))
        clips = extract_clips(vr, clip_duration, n_frames)

        game_dir = os.path.join(processed_dir, row["game_id"])
        os.makedirs(game_dir, exist_ok=True)

        for clip_idx, (start_sec, clip) in enumerate(clips):
            end_sec = start_sec + clip_duration

            clip_name = f"half{half}_clip_{clip_idx:06d}.npy"
            clip_path = os.path.join(game_dir, clip_name)

            if not os.path.exists(clip_path):
                np.save(clip_path, clip)

            clips_data.append({
                "game_id":   row["game_id"],
                "split":     row["split"],
                "half":      half,
                "start_sec": start_sec,
                "end_sec":   end_sec,
                "clip_path": clip_path,
            })

    return clips_data

def preprocess_all(games_index_path, processed_dir, clips_index_path, clip_duration=5, n_frames=10):
    """
    Processa todos os jogos do games_index.csv e escreve o clips_index.csv.

    games_index_path  — caminho pro games_index.csv
    processed_dir     — pasta raiz de data/processed/
    clips_index_path  — caminho de saída do clips_index.csv
    clip_duration     — duração de cada clip em segundos (default 5)
    n_frames          — frames por clip (default 10)
    """

    df = pd.read_csv(games_index_path)
    df = df[df["valid"] == "True"]

    os.makedirs(os.path.dirname(clips_index_path), exist_ok=True)

    all_clips = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocessing jogos"):
        try:
            clips_data = preprocess_game(row, processed_dir, clip_duration, n_frames)
            all_clips.extend(clips_data)
        except Exception as e:
            print(f"  [ERRO] {row['game_id']}: {e}")

    if not all_clips:
        print("Nenhum clip gerado.")
        return

    with open(clips_index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_clips[0].keys())
        writer.writeheader()
        writer.writerows(all_clips)

    print(f"clips_index.csv salvo em: {clips_index_path} ({len(all_clips)} clips)")

# TERMINAL

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrai clips de cada jogo e salva como tensores .npy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--games_index", default=DEFAULT_GAMES_INDEX,
        help="Caminho pro games_index.csv (default: data/labels/games_index.csv)",
    )
    parser.add_argument(
        "--processed_dir", default=DEFAULT_PROCESSED_DIR,
        help="Pasta raiz de saída dos .npy (default: data/processed/)",
    )
    parser.add_argument(
        "--clips_index", default=DEFAULT_CLIPS_INDEX,
        help="Caminho de saída do clips_index.csv (default: data/labels/clips_index.csv)",
    )
    parser.add_argument(
        "--clip_duration", type=int, default=5,
        help="Duração de cada clip em segundos (default: 5)",
    )
    parser.add_argument(
        "--n_frames", type=int, default=10,
        help="Frames a extrair por clip (default: 10)",
    )
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    preprocess_all(
        games_index_path=args.games_index,
        processed_dir=args.processed_dir,
        clips_index_path=args.clips_index,
        clip_duration=args.clip_duration,
        n_frames=args.n_frames,
    )
 
 
if __name__ == "__main__":
    main()