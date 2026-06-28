"""
build_index.py
Constrói o games_index.csv a partir dos jogos baixados.

Uso:
    python src/build_index.py
    python src/build_index.py --league epl --splits v1
    python src/build_index.py --league epl --splits train
    python src/build_index.py --output_path data/labels/games_index.csv
"""

import re
import os
import csv
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_games import (
    get_games,
    game_already_downloaded,
    LEAGUE_PREFIXES,
    DEFAULT_FILES,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOCAL_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "labels", "games_index.csv")


# --- Geração de identificadores únicos por jogo ---


def make_game_id(game_path):
    """A partir do path de um jogo constrói um id simples e legível.

    Args:
        game_path (str): Path relativo do jogo no formato SoccerNet
            (ex: 'england_epl/2016-2017/2016-12-04 - 16-30 Bournemouth 4 - 3 Liverpool').

    Returns:
        str: Identificador legível da partida
            (ex: '2016-2017_2016-12-04-16-30-bournemouth-4-3-liverpool').

    Example:
        >>> make_game_id('england_epl/2016-2017/2016-12-04 - 16-30 Bournemouth 4 - 3 Liverpool')
        '2016-2017_2016-12-04-16-30-bournemouth-4-3-liverpool'
    """
    game_id = re.sub(r" ", "-", game_path.lower())
    game_id = re.sub(r"/", "_", game_id)
    game_id = re.sub(r"-{2,}", "-", game_id)
    return "_".join(game_id.split("_")[1:])


# --- Construção do índice em CSV ---


def build_index(leagues, splits, local_dir, files, output_path):
    """Constrói um CSV com informações e caminhos para cada jogo do dataset.

    Args:
        leagues (list[str]): Códigos das ligas (ex: ['epl', 'bundesliga']).
            Devem ser chaves válidas de LEAGUE_PREFIXES.
        splits (list[str]): Lista de splits a indexar
            (ex: ['train', 'valid', 'test']).
        local_dir (str): Diretório raiz onde os jogos estão armazenados.
        files (list[str]): Lista de arquivos esperados por jogo
            (ex: ['1_224p.mkv', '2_224p.mkv', 'Labels-v2.json']).
        output_path (str): Caminho onde o CSV será salvo.

    Returns:
        None: O resultado é salvo diretamente em output_path.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for league in leagues:
        games = get_games(league, splits)

        for split, game_path in games:
            game_dir = os.path.join(local_dir, game_path)
            game_id = make_game_id(game_path)
            season = game_path.split("/")[1]

            row = {
                "game_id": game_id,
                "league": league,
                "season": season,
                "split": split,
            }

            for f in files:
                row[f] = os.path.join(game_dir, f)

            row["valid"] = game_already_downloaded(local_dir, game_path, files)

            rows.append(row)

    if not rows:
        print("Nenhum jogo encontrado.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Index salvo em: {output_path} ({len(rows)} jogos)")


def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Constrói o games_index.csv a partir dos jogos baixados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--league",
        nargs="+",
        choices=list(LEAGUE_PREFIXES.keys()),
        default=["epl"],
        help="Liga(s) a serem indexadas (default: epl)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["v1"],
        help="Splits a serem baixados: train valid test challenge v1 all (default: v1)",
    )
    parser.add_argument(
        "--local_dir",
        default=DEFAULT_LOCAL_DIR,
        help="Diretório raiz onde os jogos serão salvos (default: data/raw)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help=f"Arquivos a baixar por jogo (default: {DEFAULT_FILES})",
    )
    parser.add_argument(
        "--output_path",
        default=DEFAULT_OUTPUT_PATH,
        help="Caminho de saída do csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    build_index(
        leagues=args.league,
        local_dir=args.local_dir,
        splits=args.splits,
        files=args.files,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
