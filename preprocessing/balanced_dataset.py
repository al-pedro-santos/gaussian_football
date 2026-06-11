import numpy as np
import pandas as pd

def balanced_df(df, col_base='game_id', threshold=0.5, random_state=1):
    '''
    É esperado que o dataframe dado contenha dados apenas de um split (train, val ou test)

    Retorna um dataset balanceados entre momentos highlight e não highlights.
    O balanceamento é feito com base na partida 'game_id' por padrão
    '''
    games = df[col_base].unique()

    balanced = []

    for game in games:
        df_game = df[df[col_base] == game]
        # Highlights:
        highlights = df_game[df_game['arousal_score'] >= threshold]
        num_highlights = len(highlights)
        # Não highlights:
        non_highlights = df_game[df_game['arousal_score'] < threshold]

        # Seleciona aleatoriamente o mesmo número de amostras
        n_samples = min(num_highlights, len(non_highlights))
        sampled_non_highlights = non_highlights.sample(n=n_samples, random_state=random_state)

        # Junta highlights e não-highlights
        balanced.append(pd.concat([highlights, sampled_non_highlights], ignore_index=True))

    return pd.concat(balanced, ignore_index=True)
