import numpy as np
import pandas as pd
from pathlib import Path
import shutil


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


def balanced_df_window(
    df,
    window_col="window_id",
    clip_col="clip_id",
    label_col="label",
    score_col="arousal_score",
    highlight_name=None,
    shots_target_name=None,
    threshold=0.5,
    random_state=1,
):
    """
    O balanceamento é feito a partir das janelas: 
    - Uma janela de highlight tem k clips
    - Quero uma janela de não highlight com k clips
    
    Passaremos a considerar shots on target e off target, então o que muda na divisão é:
    - coleto as janelas highlight (quantidade g)
    - coleto as janelas com label shots on target e off target (quantidade s)
    - coleto umas quantidade g + s de janelas comuns
        (garantindo janelas a mais para se passaramos a considerar shots como highlight há teremos dados balanceados,
        mas a princípio escolher g dessas janelas comuns dará um dataset balanceado para goal vs comum)


    Regras
    -------
    - Cada label em highlight_name e shots_target_name é tratada
      como uma categoria independente.

    - Uma janela é considerada Common somente se
      max(arousal_score) < threshold.

    - Nenhum clip pode pertencer a duas janelas selecionadas.

    Returns
    -------
    balanced_df
        Dataset balanceado.

    unused_df
        Todos os clips que não foram utilizados.
    """
    if highlight_name is None:
        highlight_name = []

    if shots_target_name is None:
        shots_target_name = []

    rng = np.random.RandomState(random_state)

    event_labels = highlight_name + shots_target_name


    # Cria um resumo das janelas
    windows = []

    for window_id, group in df.groupby(window_col):
        labels = set(group[label_col])
        category = None

        # prioridade para highlight
        for lbl in highlight_name:
            if lbl in labels:
                category = lbl
                break

        # depois shots
        if category is None:
            for lbl in shots_target_name:
                if lbl in labels:
                    category = lbl
                    break

        # janela comum
        if category is None:
            if group[score_col].max() < threshold:
                category = "Common"
            else:
                continue

        windows.append({
            "window_id": window_id,
            "category": category,
            "n_clips": len(group)
        })

    windows = pd.DataFrame(windows)

    # adiciona categoria ao dataframe
    window_category = (
        windows
        .set_index(window_col)["category"]
        .to_dict()
    )

    df = df.copy()
    df["window_category"] = (
        df[window_col]
        .map(window_category)
        .fillna("Discarded")
    )

    # separa categorias
    categories = {
        lbl: windows[windows.category == lbl].copy()
        for lbl in event_labels
    }

    commons = windows[
        windows.category == "Common"
    ].sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    
    # Seleciona eventos
    used_clips = set()

    selected_event_windows = {lbl: [] for lbl in event_labels}

    n_highlight = 0
    n_shots = 0

    for lbl in event_labels:
        for _, row in categories[lbl].iterrows():
            clips = set(df.loc[df[window_col] == row.window_id, clip_col])

            if not clips.isdisjoint(used_clips):
                continue

            selected_event_windows[lbl].append(row.window_id)
            used_clips.update(clips)

            if lbl in highlight_name:
                n_highlight += 1
            else:
                n_shots += 1

    # Seleciona Common
    target_common = n_highlight + n_shots

    selected_common_windows = []

    for _, row in commons.iterrows():
        if len(selected_common_windows) >= target_common:
            break

        clips = set(
            df.loc[
                df[window_col] == row.window_id,
                clip_col
            ]
        )

        if not clips.isdisjoint(used_clips):
            continue

        selected_common_windows.append(row.window_id)
        used_clips.update(clips)

    # monta dataset final
    ordered_frames = []

    # highlights
    for lbl in highlight_name:
        for window in selected_event_windows[lbl]:
            ordered_frames.append(
                df[df[window_col] == window]
            )

    # shots
    for lbl in shots_target_name:
        for window in selected_event_windows[lbl]:
            ordered_frames.append(
                df[df[window_col] == window]
            )

    # commons
    for window in selected_common_windows:
        ordered_frames.append(df[df[window_col] == window])

    if ordered_frames:
        balanced_df = pd.concat(ordered_frames, ignore_index=True)
    else:
        balanced_df = pd.DataFrame(columns=df.columns)

    used_windows = []

    for lst in selected_event_windows.values():
        used_windows.extend(lst)

    used_windows.extend(selected_common_windows)
    unused_df = df[~df[window_col].isin(used_windows)].copy()

    return balanced_df, unused_df



# função para separar em outra pasta os clipes não usados no dataset balanceado
def move_unused_clips(
        unused_df,
        source_root,
        destination_root,
        path_col="clip_path"):
    """
    Move para outra pasta todos os clipes que não foram utilizados.

    Parameters
    - unused_df : DataFrame retornado pela função anterior
    - source_root : pasta onde estão os clipes
    - destination_root : pasta de destino
    - path_col : coluna contendo o caminho relativo ou absoluto do clip
    """
    source_root = Path(source_root)
    destination_root = Path(destination_root)

    destination_root.mkdir(parents=True, exist_ok=True)

    moved = 0

    for clip in unused_df[path_col].unique():
        src = Path(clip)

        if not src.is_absolute():
            src = source_root / clip

        if not src.exists():
            continue

        dst = destination_root / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))
        moved += 1

    print(f"{moved} arquivos movidos.")