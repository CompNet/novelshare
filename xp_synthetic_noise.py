from typing import Callable
import pathlib as pl
import functools as ft
import itertools as it
from dataclasses import dataclass
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
    make_plugin_splice,
)
from novelties_bookshare.decrypt import decrypt_tokens
from novelties_bookshare.experiments.data import load_book
from novelties_bookshare.experiments.noise import (
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
    novelties_path: str
    min_noise: float = 0.0
    max_noise: float = 0.1
    noise_step: float = 0.01
    min_hash_len: int = 64
    max_hash_len: int = 65
    jobs_nb: int = 1


@ex.automain
def main(
    _run: Run,
    novelties_path: str | pl.Path,
    min_noise: float,
    max_noise: float,
    noise_step: float,
    min_hash_len: int,
    max_hash_len: int,
    jobs_nb: int,
):
    print_config(_run)
    assert 0.0 <= min_noise < 1.0
    assert 0.0 < max_noise <= 1.0
    assert 0.0 < noise_step <= (max_noise - min_noise)
    assert 1 <= min_hash_len <= 64
    assert 2 <= max_hash_len <= 65

    if isinstance(novelties_path, str):
        novelties_path = pl.Path(novelties_path)
    novelties_path = novelties_path.expanduser()

    corpus = [
        novelties_path / "corpus" / "Brave_New_World",
        novelties_path / "corpus" / "The_Black_Company",
        novelties_path / "corpus" / "The_Blade_Itself",
        novelties_path / "corpus" / "The_Colour_Of_Magic",
        novelties_path / "corpus" / "The_Light_Fantastic",
    ]

    strategies = [
        Strategy("naive", decrypt_tokens),
        Strategy(
            "propagate",
            ft.partial(decrypt_tokens, decryption_plugins=[make_plugin_propagate()]),
        ),
        Strategy(
            "splice",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_splice(max_token_len=24, max_splits_nb=4)
                ],
            ),
        ),
        Strategy(
            "mlm",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_mlm("answerdotai/ModernBERT-base", window=16)
                ],
            ),
        ),
        Strategy(
            "splice->mlm->propagate",
            ft.partial(
                decrypt_tokens,
                decryption_plugins=[
                    make_plugin_splice(max_token_len=24, max_splits_nb=4),
                    make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
                    make_plugin_propagate(),
                ],
            ),
        ),
    ]

    noise_fns = [substitute, delete, add, ocr_scramble, token_split, token_merge]
    for noise_fn in noise_fns:
        if noise_fn == "ocr_scramble":
            _run.info[f"{noise_fn.__name__}.noise_unit"] = "WER"
        else:
            _run.info[f"{noise_fn.__name__}.noise_unit"] = "proportion"

    noise_proportions = np.arange(min_noise, max_noise, noise_step)

    hash_lens = list(range(min_hash_len, max_hash_len))

    def decrypt_setup_test(
        job_i: int,
        book_path: pl.Path,
        strategy: Strategy,
        noise_fn: Callable[[list[str], float], list[str]],
        hash_len: int,
        noise_proportion: float,
    ) -> tuple[int, float]:
        tokens, tags = load_book(book_path)
        encrypted_tokens = encrypt_tokens(tokens)
        user_tokens = noise_fn(tokens, noise_proportion)
        decrypted_tokens = strategy.decrypt_fn(
            encrypted_tokens, tags, user_tokens, hash_len
        )
        recovered_tokens = sum(
            1 if d == t else 0 for d, t in zip(decrypted_tokens, tokens)
        )
        recovered_tokens_proportion = recovered_tokens / len(tokens)
        return job_i, recovered_tokens_proportion

    setups = list(
        it.product(corpus, strategies, noise_fns, hash_lens, noise_proportions)
    )
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb, return_as="generator_unordered") as parallel:
        for job_i, recovered_tokens_proportion in parallel(  # type: ignore
            delayed(decrypt_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            book_path, strategy, noise_fn, hash_len, noise_proportion = setups[job_i]
            setup_name = f"b={book_path.name}.s={strategy.name}.n={noise_fn.__name__}.h={hash_len}"
            _run.log_scalar(
                f"{setup_name}.recovered_tokens_proportion",
                recovered_tokens_proportion,
                noise_proportion,
            )
            progress.update()
