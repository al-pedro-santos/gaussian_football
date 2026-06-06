"""
download_games.py
Baixa jogos do SoccerNet para uma liga específica.

Uso:
    python src/download_games.py --league epl --splits v1
    python src/download_games.py --league epl --splits v1 --dry_run
    python src/download_games.py --league epl --splits v1 --count_only
    python src/download_games.py --league epl --splits train valid --password SENHA
"""

import os
import argparse
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

# CONSTANTES

DEFAULT_FILES = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]
VALID_SPLITS = ["train", "valid", "test", "challenge"]

# Mapeia os códigos de liga usados neste projeto para os prefixos
# de diretório do SoccerNet. Os prefixos seguem a convenção interna
# do dataset: {país}_{liga}.
LEAGUE_PREFIXES = {
    "epl": "england_epl",
    "bundesliga": "germany_bundesliga",
    "ligue-1": "france_ligue-1",
    "ucl": "europe_uefa-champions-league",
    "serie-a": "italy_serie-a",
    "laliga": "spain_laliga",
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOCAL_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

# LIDANDO COM SPLITS


def normalize_splits(splits):
    """
    Normaliza a entrada de splits para uma lista válida.

    Args:
        splits (str | list): Split(s) desejados. Aceita 'v1', 'all',
            splits explícitos ou listas combinando esses valores.

    Returns:
        list[str]: Lista de splits válidos sem duplicatas.

    Raises:
        ValueError: Se algum split não for reconhecido.
    """
    # garante que splits seja sempre uma lista
    if not isinstance(splits, list):
        splits = [splits]

    normalized = []
    for split in splits:
        # "v1" signifca que tem os 3 splits principais (train, valid, test)
        if split == "v1":
            normalized.extend(["train", "valid", "test"])
        # "all" inclui também o split de challenge
        elif split == "all":
            normalized.extend(["train", "valid", "test", "challenge"])
        # split explícito e válido
        elif split in VALID_SPLITS:
            normalized.append(split)
        # erro de digitação
        else:
            raise ValueError(
                f"Split inválido: '{split}'. Use: {', '.join(VALID_SPLITS)} ou 'v1' / 'all'."
            )

    # remove duplicatas preservando a ordem
    seen = set()
    return [s for s in normalized if not (s in seen or seen.add(s))]


# LISTAGEM DOS JOGOS


def get_games(league, splits):
    """Retorna a lista de jogos disponíveis para uma liga e splits dados.

    Args:
        league (str): Código da liga. Deve ser uma chave válida de
            LEAGUE_PREFIXES (ex: 'epl', 'bundesliga').
        splits (str | list): Split(s) desejados. Aceita 'v1', 'all',
            splits explícitos ou listas combinando esses valores.

    Returns:
        list[tuple[str, str]]: Lista de tuplas (split, game_path), onde
            game_path é o caminho relativo do jogo no formato SoccerNet
            (ex: 'england_epl/2016-2017/2016-12-04 - ...').

    Raises:
        ValueError: Se a liga não for reconhecida.
    """
    splits = normalize_splits(splits)
    prefix = LEAGUE_PREFIXES.get(league)
    if prefix is None:
        raise ValueError(
            f"Liga desconhecida: '{league}'. Opções: {list(LEAGUE_PREFIXES.keys())}"
        )

    games = []
    for split in splits:
        for game in getListGames(split, task="spotting"):
            if game.startswith(prefix + os.sep) or game.startswith(prefix + "/"):
                games.append((split, game))

    return games


def count_games(league, splits):
    """Exibe um resumo dos jogos disponíveis por split e retorna a lista.

    Args:
        league (str): Código da liga.
        splits (str | list): Split(s) desejados.

    Returns:
        list[tuple[str, str]]: Lista de tuplas (split, game_path),
            igual ao retorno de get_games.
    """
    games = get_games(league, splits)

    split_counts = {}
    for split, _ in games:
        split_counts[split] = split_counts.get(split, 0) + 1

    print(f"\nTotal de jogos {league.upper()}: {len(games)}")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count} jogos")
    print()

    return games


# CHECAGEM DE EXISTÊNCIA


def game_already_downloaded(local_dir, game_path, files):
    """Verifica se todos os arquivos esperados de um jogo já existem no disco.

    Args:
        local_dir (str): Diretório raiz onde os jogos são armazenados.
        game_path (str): Caminho relativo do jogo no formato SoccerNet.
        files (list[str]): Lista de nomes de arquivos esperados.

    Returns:
        bool: True se todos os arquivos existem, False caso contrário.
    """
    game_dir = os.path.join(local_dir, game_path)
    for f in files:
        if not os.path.isfile(os.path.join(game_dir, f)):
            return False
    return True


# DOWNLOADER


def download_games(league, local_dir, password, splits, files, dry_run=False):
    """Baixa os arquivos das partidas para a liga e splits especificados.

    Jogos cujos arquivos já existem no disco são pulados automaticamente.
    Se dry_run=True, lista o que seria baixado sem efetuar nenhum download.

    Args:
        league (str): Código da liga.
        local_dir (str): Diretório raiz onde os arquivos serão salvos.
        password (str | None): Senha SoccerNet para arquivos protegidos.
        splits (str | list): Split(s) desejados.
        files (list[str]): Lista de arquivos a baixar por jogo.
        dry_run (bool): Se True, apenas lista as ações sem executá-las.
            Default: False.

    Raises:
        RuntimeError: Se nenhum jogo for encontrado para a liga e splits dados.
    """
    games = get_games(league, splits)
    if not games:
        raise RuntimeError(
            f"Nenhum jogo encontrado para liga='{league}', splits={splits}."
        )

    active_splits = sorted(set(s for s, _ in games))
    print(
        f"{'[DRY RUN] ' if dry_run else ''}Processando {len(games)} jogos "
        f"{league.upper()} | splits: {', '.join(active_splits)}"
    )
    print(f"Arquivos alvo: {files}\n")

    if dry_run:
        for split, game in games:
            status = (
                "ok" if game_already_downloaded(local_dir, game, files) else "baixar"
            )
            print(f"  [{status:6s}] {split}: {game}")
        return

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    if password:
        downloader.password = password

    skipped, downloaded, failed = 0, 0, 0

    for split, game in games:
        if game_already_downloaded(local_dir, game, files):
            print(f"  [skip]  {split}: {game}")
            skipped += 1
            continue

        print(f"  [down]  {split}: {game}")
        try:
            downloader.downloadGame(game=game, files=files, spl=split, verbose=False)
            downloaded += 1
        except Exception as e:
            print(f"  [ERRO]  {game}: {e}")
            failed += 1

    print(f"\nConcluído. baixados={downloaded}  pulados={skipped}  erros={failed}")


# RODAR NO TERMINAL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baixa jogos do SoccerNet para uma liga específica.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--league",
        choices=list(LEAGUE_PREFIXES.keys()),
        default="epl",
        help="Liga a ser baixada (default: epl)",
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
        "--password",
        default=None,
        help="Senha SoccerNet (necessária para vídeos)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help=f"Arquivos a baixar por jogo (default: {DEFAULT_FILES})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Lista os jogos que seriam baixados sem baixar nada",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Apenas conta os jogos disponíveis e sai",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.count_only:
        count_games(args.league, args.splits)
        return

    download_games(
        league=args.league,
        local_dir=args.local_dir,
        password=args.password,
        splits=args.splits,
        files=args.files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
