from copy import error
from typing import Callable, Literal, Optional
import time
import pathlib as pl
import functools as ft
import itertools as it
from dataclasses import dataclass
from more_itertools import flatten
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
    make_plugin_case,
    make_plugin_cycle,
)
from novelties_bookshare.decrypt import decrypt_tokens
from novelties_bookshare.experiments.data import iter_book_chapters
from novelties_bookshare.experiments.metrics import record_decryption_metrics_
from novelties_bookshare.experiments.errors import (
    substitute,
    delete,
    add,
    token_split,
    token_merge,
)

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))

DecryptFn = Callable[
    # args:
    [
        # encrypted_chapters
        list[list[str]],
        # user_chapters
        list[list[str]],
        # hash_len
        int | None,
    ],
    # returns: decrypted tokens
    list[str],
]


@dataclass
class Strategy:
    name: str
    decrypt_fn: DecryptFn


@ex.config
def config():
    min_error_ratio: float
    max_error_ratio: float
    error_ratio_step: float
    hash_len: int = 64
    chapter_limit: Optional[int] = None
    jobs_nb: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"


@ex.automain
def main(
    _run: Run,
    min_error_ratio: float,
    max_error_ratio: float,
    error_ratio_step: float,
    hash_len: int,
    chapter_limit: Optional[int],
    jobs_nb: int,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert min_error_ratio >= 0
    assert max_error_ratio > min_error_ratio
    assert hash_len > 0 and hash_len <= 64

    corpus = [
        pl.Path("./data/Frankenstein/PG84/"),
        pl.Path("./data/Moby_Dick/PG15/"),
        pl.Path("./data/PrideAndPrejudice/PG1342"),
    ]

    strategies = [
        Strategy("naive", decrypt_tokens),
        Strategy(
            "propagate",
            ft.partial(decrypt_tokens, decryption_plugins=[make_plugin_propagate()]),
        ),
        Strategy(
            "split",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_split(max_token_len=24, max_splits_nb=4)
                ],
            ),
        ),
        Strategy(
            "bert",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    )
                ],
            ),
        ),
        Strategy(
            "pipe",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_propagate(),
                    make_plugin_case(),
                    make_plugin_split(max_token_len=24, max_splits_nb=4),
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    ),
                ],
            ),
        ),
        Strategy(
            "cycle",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_cycle(
                        [
                            make_plugin_propagate(),
                            make_plugin_case(),
                            make_plugin_split(max_token_len=24, max_splits_nb=4),
                            make_plugin_mlm(
                                "answerdotai/ModernBERT-base", window=16, device=device
                            ),
                        ],
                        budget=None,
                    )
                ],
            ),
        ),
    ]

    errors_fns = [substitute, delete, add, token_split, token_merge]
    for errors_fn in errors_fns:
        _run.info[f"{errors_fn.__name__}.errors_unit"] = "Ratio of syntethic errors"

    error_ratio = [
        float(i) for i in np.arange(min_error_ratio, max_error_ratio, error_ratio_step)
    ]

    def decrypt_setup_test(
        job_i: int,
        book_path: pl.Path,
        strategy: Strategy,
        errors_fn: Callable[[list[str], int], list[str]],
        error_ratio: float,
    ) -> tuple[int, list[list[str]], list[str], float]:
        t0 = time.process_time()
        chapters = list(iter_book_chapters(book_path, chapter_limit=chapter_limit))
        encrypted_chapters = [
            encrypt_tokens(chapter, hash_len=hash_len) for chapter in chapters
        ]
        user_chapters = [
            errors_fn(chapter, int(len(chapter) * error_ratio)) for chapter in chapters
        ]
        decrypted_tokens = strategy.decrypt_fn(
            encrypted_chapters, user_chapters, hash_len
        )
        t1 = time.process_time()
        return job_i, chapters, decrypted_tokens, t1 - t0

    setups = list(it.product(corpus, strategies, errors_fns, error_ratio))
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb) as parallel:
        for job_i, gold_chapters, decrypted_tokens, duration_s in parallel(
            delayed(decrypt_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            gold_tokens = list(flatten(gold_chapters))
            book_path, strategy, errors_fn, error_ratio = setups[job_i]
            setup_name = f"b={book_path.name}.s={strategy.name}.n={errors_fn.__name__}"
            record_decryption_metrics_(
                _run, setup_name, gold_tokens, decrypted_tokens, duration_s
            )
            progress.update()
