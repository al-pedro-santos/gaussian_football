import pandas as pd
import csv

with open("data/labels/games_index.csv") as f:
    index = [row["game_id"] for row in csv.DictReader(f) if row["valid"] == "True"]

df = pd.read_csv("data/labels/labels_all.csv")
processados = set(df["game_id"].unique())

faltando = [g for g in index if g not in processados]
print(f"Faltando {len(faltando)} jogos:")
for g in faltando:
    print(f"  {g}")
