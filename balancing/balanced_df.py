import pandas as pd
import numpy as np


def _extract_half(clip_path):
    """Extrai 'half_<n>' do caminho do clip. Retorna None se nao houver."""
    return clip_path.str.extract(r'(half_\d+)', expand=False)


def _extract_clip_num(clip_path):
    """Extrai o numero inteiro de 'clip_<numero>.mp4' do caminho do clip."""
    return clip_path.str.extract(r'clip_(\d+)\.mp4', expand=False).astype(int)


def balanced_df_window(
    df,
    clip_col="clip_id",
    label_col="label",
    event_id_col="event_id",
    score_col="arousal_score",
    game_col="game_id",
    highlight_names=None,
    shot_names=None,
    category_map=None,
    background_label="Background",
    threshold=0.5,
    max_trials=1000,
    random_state=1,
):
    """
    Constroi um dataset balanceado para treino de Margin Ranking Loss, em que
    cada janela de referencia (Goal, Shot on target, Shot off target, etc.) e
    pareada com uma janela de Background do MESMO TAMANHO exato, pertencente
    ao mesmo (game_id, half).

    Janela de referencia
    ---------------------
    Formada por todos os clips com o mesmo (game_col, half, event_id_col).
    E uma unidade indivisivel: nunca e cortada, dividida ou alterada.

    Janela de Background
    ----------------------
    Sequencia de k clips CONSECUTIVOS (mesmo game_col e half), com label
    igual a background_label, todos com arousal_score < threshold, e que nao
    reutiliza nenhum clip ja usado por outra janela Background selecionada.
    E encontrada por BUSCA ALEATORIA (nao enumeracao exaustiva): a cada
    tentativa, escolhe-se aleatoriamente um possivel clip inicial dentro do
    mesmo (game_col, half) e testa-se se a janela de k clips a partir dali
    satisfaz todos os criterios de validacao. Ate max_trials tentativas por
    janela de referencia; se nenhuma for valida, o evento fica sem par
    (matched=False) mas continua no summary.

    Ordem de processamento
    ------------------------
    As janelas de referencia sao processadas da MAIOR para a MENOR (em numero
    de clips), o que aumenta a chance de encontrar janelas Background validas
    para os tamanhos mais raros/dificeis antes que o espaco disponivel seja
    consumido por janelas menores.

    Parameters
    ----------
    df : pd.DataFrame
        Um clip por linha.
    clip_col : str
        Coluna com o caminho do clip (deve conter 'half_<n>' e
        'clip_<numero>.mp4' em algum ponto da string).
    label_col : str
        Coluna com o tipo do evento (ex: 'type'). Usada para identificar
        Background e para mapear a categoria final (window_category) via
        category_map.
    event_id_col : str
        Coluna com o identificador do evento (ex: 'event_id'), unico dentro
        de (game_col, half) para eventos de highlight/shot (ex: 'goal_3').
        Para Background, o valor desta coluna e ignorado -- janelas de
        Background sao formadas por busca aleatoria de blocos consecutivos
        de label_col == background_label, nao por agrupamento de
        event_id_col.
    score_col : str
        Coluna com o arousal_score.
    game_col : str
        Coluna que identifica o jogo/partida.
    highlight_names : list[str] or None
        Valores de label_col tratados como highlight (ex: ['goal']).
    shot_names : list[str] or None
        Valores de label_col tratados como shot (ex: ['shot']). Pode conter
        mais de um valor se houver distincao on/off target no proprio
        label_col (ex: ['shot_on_target', 'shot_off_target']).
    category_map : dict or None
        Mapeia valores de label_col para o nome de categoria usado no
        window_category/summary_df (ex: {'goal': 'Goal', 'shot': 'Shot on
        target'}). Se None, usa o proprio valor de label_col como categoria.
    background_label : str
        Valor de label_col que identifica clips de fundo (ex: 'Background').
    threshold : float
        Limiar de arousal_score abaixo do qual um clip Background e
        elegivel para compor uma janela Background.
    max_trials : int
        Numero maximo de tentativas aleatorias de janela Background por
        janela de referencia, antes de desistir e marcar matched=False.
    random_state : int
        Seed para reprodutibilidade da busca aleatoria e da ordem de
        processamento.

    Returns
    -------
    balanced_df : pd.DataFrame
        Todas as janelas de referencia (integrais) + todas as janelas
        Background selecionadas. Todas as colunas originais de df sao
        preservadas, mais a coluna 'window_category'. Ordenado por
        (game_col, half, clip_num).
    unused_df : pd.DataFrame
        Clips que nao pertencem a nenhuma janela de referencia nem a nenhuma
        janela Background selecionada. Preserva todas as colunas originais
        (sem window_category).
    summary_df : pd.DataFrame
        Uma linha por categoria (incluindo 'Background') com n_windows e
        n_clips. Inclui tambem uma linha 'TOTAL' e duas entradas escalares
        (expostas via summary_df.attrs) 'matched_windows' e
        'unmatched_windows'.
    """
    if highlight_names is None:
        highlight_names = []
    if shot_names is None:
        shot_names = []
    event_labels = list(highlight_names) + list(shot_names)

    if category_map is None:
        category_map = {}

    def category_of(label):
        return category_map.get(label, label)

    rng = np.random.RandomState(random_state)

    # --- Pre-processamento: extrair half, clip_num, ordenar ---
    df = df.copy()
    df['_half'] = _extract_half(df[clip_col])
    df['_clip_num'] = _extract_clip_num(df[clip_col])
    has_half = df['_half'].notna().any()
    if not has_half:
        df['_half'] = '_NA_'

    df = df.sort_values(by=[game_col, '_half', '_clip_num']).reset_index(drop=True)

    # --- Construcao das janelas de referencia ---
    is_event = df[label_col].isin(event_labels)
    event_df = df[is_event]

    reference_windows = []
    if len(event_df) > 0:
        grouped = event_df.groupby([game_col, '_half', event_id_col], sort=False)
        for (game_id, half, event_id), group in grouped:
            label = group[label_col].iloc[0]
            indices = group.index.values
            k = len(indices)
            reference_windows.append({
                "game_id": game_id,
                "half": half,
                "event_id": event_id,
                "label": label,
                "category": category_of(label),
                "k": k,
                "indices": indices,
            })

    # Processa da maior para a menor janela, para maximizar chance de achar
    # Background valido para os tamanhos mais dificeis primeiro.
    reference_windows.sort(key=lambda w: w["k"], reverse=True)

    # --- Estruturas auxiliares ---
    used_indices = set()  # clips usados em janelas Background selecionadas

    # Pre-indexa, para cada (game_id, half), os arrays de indice, clip_num e
    # arousal_score dos clips de Background, ja ordenados por clip_num (a
    # ordenacao global de df garante isso). Tambem mapeia clip_num -> posicao
    # no array, para checar continuidade rapidamente.
    bg_mask = df[label_col] == background_label
    bg_by_group = {}
    for (game_id, half), group in df[bg_mask].groupby([game_col, '_half'], sort=False):
        idx = group.index.values
        clip_nums = group['_clip_num'].values
        scores = group[score_col].values
        bg_by_group[(game_id, half)] = {
            "indices": idx,
            "clip_num": clip_nums,
            "score": scores,
            # mapa clip_num -> posicao no array (para localizar rapidamente
            # se um certo clip_num inicial existe e onde)
            "pos_of_clipnum": {cn: i for i, cn in enumerate(clip_nums)},
        }

    matched_count = 0
    unmatched_count = 0
    unmatched_events = []

    selected_background_windows = []  # lista de arrays de indices

    for ref in reference_windows:
        game_id, half, k = ref["game_id"], ref["half"], ref["k"]
        bucket = bg_by_group.get((game_id, half))

        found = False
        if bucket is not None and len(bucket["indices"]) >= k:
            clip_num = bucket["clip_num"]
            score = bucket["score"]
            indices = bucket["indices"]
            pos_of_clipnum = bucket["pos_of_clipnum"]
            n_bg = len(indices)

            # Candidatos a clip inicial: qualquer clip_num de Background que
            # tenha pelo menos k-1 clip_nums consecutivos depois dele dentro
            # do array de Background (nao necessariamente no df inteiro --
            # a checagem real de continuidade e feita na validacao abaixo).
            # Aqui so usamos a posicao no array de Background como ponto de
            # partida da busca aleatoria.
            if n_bg >= k:
                candidate_positions = np.arange(0, n_bg - k + 1)
            else:
                candidate_positions = np.array([], dtype=int)

            if len(candidate_positions) > 0:
                trial_positions = rng.choice(
                    candidate_positions,
                    size=min(max_trials, len(candidate_positions) * 5),
                    replace=True,
                )

                for start_pos in trial_positions:
                    end_pos = start_pos + k  # exclusivo
                    if end_pos > n_bg:
                        continue

                    window_clip_nums = clip_num[start_pos:end_pos]
                    window_indices = indices[start_pos:end_pos]
                    window_scores = score[start_pos:end_pos]

                    # Criterio 1: exatamente k clips (garantido pela fatia)
                    if len(window_indices) != k:
                        continue

                    # Criterio 2: clips consecutivos (clip_num.diff()==1)
                    diffs = np.diff(window_clip_nums)
                    if not np.all(diffs == 1):
                        continue

                    # Criterio 3: todos label == Background
                    # (garantido pela construcao do bucket, que so contem
                    # indices com label_col == background_label)

                    # Criterio 4: arousal_score < threshold para todos
                    if window_scores.max() >= threshold:
                        continue

                    # Criterio 5: nenhum indice em used_indices
                    if any(i in used_indices for i in window_indices):
                        continue

                    # Janela valida
                    used_indices.update(window_indices.tolist())
                    selected_background_windows.append(window_indices)
                    found = True
                    break

        if found:
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_events.append({
                "game_id": game_id,
                "event_id": ref["event_id"],
                "category": ref["category"],
                "matched": False,
            })

    # Construcao do dataset final
    ref_indices_all = [w["indices"] for w in reference_windows]
    bg_indices_all = selected_background_windows

    all_used_indices = []
    window_category_map = {}  # indice -> categoria

    for w in reference_windows:
        for i in w["indices"]:
            window_category_map[i] = w["category"]
        all_used_indices.append(w["indices"])

    for idx_arr in bg_indices_all:
        for i in idx_arr:
            window_category_map[i] = category_of(background_label)
        all_used_indices.append(idx_arr)

    if all_used_indices:
        all_idx = np.concatenate(all_used_indices)
    else:
        all_idx = np.array([], dtype=df.index.dtype)

    balanced_df = df.loc[all_idx].copy()
    balanced_df["window_category"] = balanced_df.index.map(window_category_map)
    balanced_df = balanced_df.sort_values(
        by=[game_col, "_half", "_clip_num"]
    ).reset_index(drop=True)
    balanced_df = balanced_df.drop(columns=["_half", "_clip_num"])

    used_idx_set = pd.Index(all_idx)
    unused_df = df.loc[~df.index.isin(used_idx_set)].copy()
    unused_df = unused_df.drop(columns=["_half", "_clip_num"]).reset_index(drop=True)

    # summary_df
    summary_rows = []

    cats_seen_order = []
    for w in reference_windows:
        if w["category"] not in cats_seen_order:
            cats_seen_order.append(w["category"])
    # mantem ordem de aparicao na lista original highlight_names + shot_names
    ordered_cats = []
    for lbl in event_labels:
        c = category_of(lbl)
        if c not in ordered_cats and c in cats_seen_order:
            ordered_cats.append(c)
    for c in cats_seen_order:
        if c not in ordered_cats:
            ordered_cats.append(c)

    for cat in ordered_cats:
        wins = [w for w in reference_windows if w["category"] == cat]
        n_windows = len(wins)
        n_clips = sum(w["k"] for w in wins)
        summary_rows.append({"category": cat, "n_windows": n_windows, "n_clips": n_clips})

    bg_cat = category_of(background_label)
    n_win_bg = len(bg_indices_all)
    n_clp_bg = sum(len(a) for a in bg_indices_all)
    summary_rows.append({"category": bg_cat, "n_windows": n_win_bg, "n_clips": n_clp_bg})

    total_windows = sum(r["n_windows"] for r in summary_rows)
    total_clips = sum(r["n_clips"] for r in summary_rows)
    summary_rows.append({"category": "TOTAL", "n_windows": total_windows, "n_clips": total_clips})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.attrs["matched_windows"] = matched_count
    summary_df.attrs["unmatched_windows"] = unmatched_count
    summary_df.attrs["unmatched_events"] = unmatched_events

    return balanced_df, unused_df, summary_df


if __name__ == "__main__":
    pass