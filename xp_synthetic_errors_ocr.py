from typing import Callable, Literal, Optional
import time
import pathlib as pl
import functools as ft
import itertools as it
from dataclasses import dataclass
from more_itertools import flatten
from tqdm import tqdm
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from novelties_bookshare.hash import hash_tokens
from novelties_bookshare.align import (
    align_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_retokenize,
    make_plugin_case,
)
from novelties_bookshare.experiments.data import iter_book_chapters, load_book
from novelties_bookshare.experiments.metrics import record_alignment_metrics_
from novelties_bookshare.experiments.errors import ocr_scramble

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))

AlignFn = Callable[
    # args:
    [
        # hashed_chapters
        list[list[str]],
        # user_chapters
        list[list[str]],
        # hash_len
        int | None,
    ],
    # returns: aligned tokens
    list[str],
]


@dataclass
class Strategy:
    name: str
    align_fn: AlignFn


@ex.config
def config():
    wer_grid: list[float]
    cer_grid: list[float]
    hash_len: int = 64
    chapter_limit: Optional[int] = None
    jobs_nb: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"


@ex.automain
def main(
    _run: Run,
    wer_grid: list[float],
    cer_grid: list[float],
    hash_len: int,
    chapter_limit: Optional[int],
    jobs_nb: int,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert hash_len > 0 and hash_len <= 64
    assert len(wer_grid) == len(cer_grid)

    corpus = [
        pl.Path("./data/Frankenstein/F-1818/"),
        pl.Path("./data/Moby_Dick/MD-1851-US/"),
        pl.Path("./data/Pride_and_Prejudice/PP-1813/"),
    ]

    strategies = [
        Strategy("naive", align_tokens),
        Strategy(
            "case", ft.partial(align_tokens, alignment_plugins=[make_plugin_case()])
        ),
        Strategy(
            "propagate",
            ft.partial(align_tokens, alignment_plugins=[make_plugin_propagate()]),
        ),
        Strategy(
            "retokenize",
            ft.partial(
                align_tokens,
                alignment_plugins=[
                    make_plugin_retokenize(max_token_len=24, max_splits_nb=4)
                ],
            ),
        ),
        Strategy(
            "mlm",
            ft.partial(
                align_tokens,
                alignment_plugins=[
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    )
                ],
            ),
        ),
        Strategy(
            "pipe",
            ft.partial(
                align_tokens,
                alignment_plugins=[
                    make_plugin_propagate(),
                    make_plugin_case(),
                    make_plugin_retokenize(max_token_len=24, max_splits_nb=4),
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    ),
                ],
            ),
        ),
    ]

    _run.info[f"ocr_scramble.errors_unit"] = "(WER,CER)"

    def align_setup_test(
        job_i: int,
        book_path: pl.Path,
        strategy: Strategy,
        wer_cer: tuple[float, float],
    ) -> tuple[int, list[list[str]], list[str], float]:
        t0 = time.process_time()
        chapters = list(iter_book_chapters(book_path, chapter_limit=chapter_limit))
        hashed_chapters = [
            hash_tokens(chapter, hash_len=hash_len) for chapter in chapters
        ]
        user_chapters = [ocr_scramble(chapter, *wer_cer) for chapter in chapters]
        aligned_tokens = strategy.align_fn(hashed_chapters, user_chapters, hash_len)
        t1 = time.process_time()
        return job_i, chapters, aligned_tokens, t1 - t0

    setups = list(it.product(corpus, strategies, zip(wer_grid, cer_grid)))
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb) as parallel:
        for job_i, gold_chapters, aligned_tokens, duration_s in parallel(
            delayed(align_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            gold_tokens = list(flatten(gold_chapters))
            book_path, strategy, (wer, cer) = setups[job_i]
            setup_name = f"b={book_path.name}.s={strategy.name}.n=ocr_scramble"
            record_alignment_metrics_(
                _run, setup_name, gold_tokens, aligned_tokens, duration_s
            )
            progress.update()
