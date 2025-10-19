from typing import Optional
import time
from more_itertools import flatten
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens, make_plugin_split
from novelties_bookshare.experiments.data import iter_book_chapters, normalize_
from novelties_bookshare.experiments.metrics import record_decryption_metrics_

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))

EDITION_SETS = {
    "Brave_New_World": {
        "Novelties": "./data/editions_diff/Brave_New_World/Novelties",
        "HC98": "./data/editions_diff/Brave_New_World/HC98",
        "HC06": "./data/editions_diff/Brave_New_World/HC06",
        "HC04": "./data/editions_diff/Brave_New_World/HC04",
        "RB06": "./data/editions_diff/Brave_New_World/RB06",
    },
    "Moby_Dick": {
        "Novelties": "./data/editions_diff/Moby_Dick/Novelties",
        "PG15": "./data/editions_diff/Moby_Dick/PG15",
        "PG2489": "./data/editions_diff/Moby_Dick/PG2489",
        "PG2701": "./data/editions_diff/Moby_Dick/PG2701",
    },
}


@ex.config
def config():
    edition_set: str
    max_token_len_range: list[int]
    max_splits_nb_range: list[int]
    hash_len: int = 64
    chapter_limit: Optional[int] = None


@ex.automain
def main(
    _run: Run,
    edition_set: str,
    max_token_len_range: list[int],
    max_splits_nb_range: list[int],
    hash_len: int,
    chapter_limit: Optional[int],
):
    print_config(_run)
    assert edition_set in EDITION_SETS
    assert hash_len > 0 and hash_len <= 64

    novelties_chapters = list(
        iter_book_chapters(
            EDITION_SETS[edition_set]["Novelties"], chapter_limit=chapter_limit
        )
    )

    wild_editions = {
        key: list(iter_book_chapters(path, chapter_limit=chapter_limit))
        for key, path in EDITION_SETS[edition_set].items()
        if key != "Novelties"
    }

    normalize_(novelties_chapters)
    for chapters in wild_editions.values():
        normalize_(chapters)

    novelties_encrypted = [
        encrypt_tokens(chapter, hash_len=hash_len) for chapter in novelties_chapters
    ]

    progress = tqdm(
        total=len(wild_editions) * len(max_token_len_range) * len(max_splits_nb_range),
        ascii=True,
    )

    for edition, user_tokens in wild_editions.items():
        for max_token_len in max_token_len_range:
            for max_split_nb in max_splits_nb_range:
                progress.set_description(
                    f"{edition}.t={max_token_len}.s={max_split_nb}"
                )

                split = make_plugin_split(max_token_len, max_split_nb)
                t0 = time.process_time()
                decrypted_tokens = decrypt_tokens(
                    novelties_encrypted,
                    user_tokens,
                    hash_len=hash_len,
                    decryption_plugins=[split],
                )
                t1 = time.process_time()

                novelties_tokens = list(flatten(novelties_chapters))
                setup_name = f"t={max_token_len}.s={max_split_nb}.e={edition}"
                record_decryption_metrics_(
                    _run,
                    setup_name,
                    novelties_tokens,
                    decrypted_tokens,
                    t1 - t0,
                )

                progress.update()
