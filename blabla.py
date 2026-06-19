import pandas as pd
from pathlib import Path

df = pd.read_csv(TEST_PATH)  # roda pra train/valid também se quiser comparar

df["clip_exists"] = df["clip_path"].apply(lambda p: Path(p).exists())
df["mel_exists"] = df["mel_path"].apply(lambda p: Path(p).exists())

missing = df[~(df["clip_exists"] & df["mel_exists"])]

print(missing["game_id"].value_counts())