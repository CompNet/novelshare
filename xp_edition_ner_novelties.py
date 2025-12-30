from typing import Literal, Optional, Generator
import time, re
import pathlib as pl
from more_itertools import flatten
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from novelties_bookshare.conll import load_conll2002_bio
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
    make_plugin_case,
)
from novelties_bookshare.experiments.data import normalize_, iter_book_chapters
from novelties_bookshare.experiments.metrics import record_decryption_metrics_, errors

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))


def iter_book_chapters_and_tags(
    path: pl.Path | str, chapter_limit: Optional[int] = None
) -> Generator[tuple[list[str], list[str]], None, None]:
    """A specialized version of :func:`.iter_book_chapters` that also yield chapter tags."""
    if isinstance(path, str):
        path = pl.Path(path)
    path = path.expanduser()

    chapter_paths = path.glob("chapter_*.conll")
    chapter_paths = sorted(
        chapter_paths,
        key=lambda p: int(re.match(r"chapter_([0-9]+)\.conll", str(p.name)).group(1)),
    )
    if chapter_paths is not None:
        chapter_paths = chapter_paths[:chapter_limit]

    for path in chapter_paths:
        chapter_tokens, chapter_tags = load_conll2002_bio(str(path))
        yield chapter_tokens, chapter_tags


@ex.config
def config():
    hash_len: int = 64
    chapter_limit: Optional[int] = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    split_max_token_len: int = 16
    split_max_splits_nb: int = 8
    mlm_window: int = 32


@ex.automain
def main(
    _run: Run,
    hash_len: int,
    chapter_limit: Optional[int],
    device: Literal["auto", "cuda", "cpu"],
    split_max_token_len: int,
    split_max_splits_nb: int,
    mlm_window: int,
):
    print_config(_run)
    assert hash_len > 0 and hash_len <= 64

    reference_chapters_and_tags = list(
        iter_book_chapters_and_tags(
            "./data/Moby_Dick/MB-Novelties", chapter_limit=chapter_limit
        ),
    )
    reference_chapters = [chapter for chapter, _ in reference_chapters_and_tags]
    reference_tags = [chapter_tags for _, chapter_tags in reference_chapters_and_tags]

    user_chapters = list(
        iter_book_chapters("./data/Moby_Dick/MB-1988", chapter_limit=chapter_limit)
    )

    normalize_(reference_chapters)
    normalize_(user_chapters)

    strategies = {
        "naive": None,
        "case": [make_plugin_case()],
        "propagate": [make_plugin_propagate()],
        "split": [
            make_plugin_split(
                max_token_len=split_max_token_len, max_splits_nb=split_max_splits_nb
            )
        ],
        "mlm": [
            make_plugin_mlm(
                "answerdotai/ModernBERT-base", window=mlm_window, device=device
            )
        ],
        "pipe": [
            make_plugin_split(
                max_token_len=split_max_token_len, max_splits_nb=split_max_splits_nb
            ),
            make_plugin_mlm(
                "answerdotai/ModernBERT-base", window=mlm_window, device=device
            ),
            make_plugin_case(),
            make_plugin_propagate(),
        ],
    }
    # { strategy => { ref_token => [ incorrect pred tokens ] } }
    all_errors = {strat: {} for strat in strategies.keys()}

    reference_encrypted = [
        encrypt_tokens(chapter, hash_len=hash_len) for chapter in reference_chapters
    ]

    progress = tqdm(strategies.items(), ascii=True, total=len(strategies))
    for strat, strat_plugins in progress:
        progress.set_description(strat)

        t0 = time.process_time()
        decrypted_tokens = decrypt_tokens(
            reference_encrypted,
            user_chapters,
            hash_len=hash_len,
            decryption_plugins=strat_plugins,
        )
        t1 = time.process_time()

        reference_tokens = list(flatten(reference_chapters))
        setup_name = f"s={strat}.e=Moby_Dick,MB-1851-US"
        record_decryption_metrics_(
            _run,
            setup_name,
            reference_tokens,
            decrypted_tokens,
            t1 - t0,
            ref_tags=list(flatten(reference_tags)),
        )
        all_errors[strat] = errors(reference_tokens, decrypted_tokens)

        progress.update()

    _run.info["errors"] = all_errors
