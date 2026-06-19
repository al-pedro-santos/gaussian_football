import pandas as pd
import os

index = pd.read_csv('/mnt/storage_C4/gaussian_football/data/labels/games_index.csv')

ok, faltando = 0, []
for _, row in index[index['split'] == 'test'].iterrows():
    arquivos = [row['1_224p.mkv'], row['2_224p.mkv'], row['Labels-v2.json']]
    if all(os.path.isfile(f) for f in arquivos):
        ok += 1
    else:
        faltando.append(row['game_id'])

print(f'OK: {ok}/18')
if faltando:
    print('Faltando:')
    for g in faltando:
        print(f'  {g}')