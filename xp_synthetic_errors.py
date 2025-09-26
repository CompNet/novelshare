from typing import Callable, Literal
import time
import pathlib as pl
import functools as ft
import itertools as it
from dataclasses import dataclass
from tqdm import tqdm
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
from novelties_bookshare.experiments.data import load_book
from novelties_bookshare.experiments.metrics import record_decryption_metrics_
from novelties_bookshare.experiments.errors import (
    substitute,
    delete,
    add,
    ocr_scramble,
    token_split,
    token_merge,
)

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))

DecryptFn = Callable[
    # args:
    [
        # encrypted_tokens
        list,
        # tags
        list[str],
        # user_tokens
        list[str],
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
    min_errors: int
    max_errors: int
    errors_step: int
    min_hash_len: int = 64
    max_hash_len: int = 65
    jobs_nb: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"


@ex.automain
def main(
    _run: Run,
    min_errors: int,
    max_errors: int,
    errors_step: int,
    min_hash_len: int,
    max_hash_len: int,
    jobs_nb: int,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert min_errors >= 0
    assert max_errors > min_errors
    assert 1 <= min_hash_len <= 64
    assert 2 <= max_hash_len <= 65

    corpus = [pl.Path("./data/editions_diff/Moby_Dick/Novelties")]

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
                    make_plugin_case(),
                    make_plugin_split(max_token_len=24, max_splits_nb=4),
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    ),
                    make_plugin_propagate(),
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
                            make_plugin_case(),
                            make_plugin_split(max_token_len=24, max_splits_nb=4),
                            make_plugin_mlm(
                                "answerdotai/ModernBERT-base", window=16, device=device
                            ),
                            make_plugin_propagate(),
                        ],
                        budget=None,
                    )
                ],
            ),
        ),
    ]

    # errors_fns = [substitute, delete, add, ocr_scramble, token_split, token_merge]
    print("/!\\ warning /!\\ ocr_scramble deactivated for now")
    errors_fns = [substitute, delete, add, token_split, token_merge]
    for errors_fn in errors_fns:
        if errors_fn == ocr_scramble:
            _run.info[f"{errors_fn.__name__}.errors_unit"] = "WER"
        else:
            _run.info[f"{errors_fn.__name__}.errors_unit"] = "proportion"

    nb_errors = list(range(min_errors, max_errors, errors_step))

    hash_lens = list(range(min_hash_len, max_hash_len))

    def decrypt_setup_test(
        job_i: int,
        book_path: pl.Path,
        strategy: Strategy,
        errors_fn: Callable[[list[str], float], list[str]],
        hash_len: int,
        nb_errors: float,
    ) -> tuple[int, list[str], list[str], list[str], float]:
        t0 = time.process_time()
        tokens, tags = load_book(book_path)
        encrypted_tokens = encrypt_tokens(tokens)
        user_tokens = errors_fn(tokens, nb_errors)
        decrypted_tokens = strategy.decrypt_fn(
            encrypted_tokens, tags, user_tokens, hash_len
        )
        t1 = time.process_time()
        return job_i, tokens, decrypted_tokens, tags, t1 - t0

    setups = list(it.product(corpus, strategies, errors_fns, hash_lens, nb_errors))
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb, return_as="generator_unordered") as parallel:
        for job_i, gold_tokens, decrypted_tokens, gold_tags, duration_s in parallel(
            delayed(decrypt_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            book_path, strategy, errors_fn, hash_len, nb_errors = setups[job_i]
            setup_name = f"b={book_path.name}.s={strategy.name}.n={errors_fn.__name__}.h={hash_len}"
            record_decryption_metrics_(
                _run, setup_name, gold_tokens, decrypted_tokens, gold_tags, duration_s
            )
            progress.update()
