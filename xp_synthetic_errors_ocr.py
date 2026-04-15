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
from novelties_bookshare.experiments.metrics import log_alignment_metrics_
from novelties_bookshare.experiments.errors import ocr_scramble

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))
from novelties_bookshare.experiments.data import (
    load_corpus,
    CorpusID,
    Strategy,
    Document,
)


@ex.config
def config():
    wer_grid: list[float]
    cer_grid: list[float]
    hash_len: int = 64
    corpus_id: CorpusID = "3novels"
    # only used when corpus_id == '3novels'
    chapter_limit: Optional[int] = None
    jobs_nb: int = 1
    device: Literal["auto", "cuda", "cpu"] = "auto"


@ex.automain
def main(
    _run: Run,
    wer_grid: list[float],
    cer_grid: list[float],
    hash_len: int,
    corpus_id: CorpusID,
    chapter_limit: Optional[int],
    jobs_nb: int,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert hash_len > 0 and hash_len <= 64
    assert len(wer_grid) == len(cer_grid)

    corpus = load_corpus(corpus_id, chapter_limit=chapter_limit)

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
                    make_plugin_retokenize(max_token_len=16, max_splits_nb=8)
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
                    make_plugin_retokenize(max_token_len=16, max_splits_nb=8),
                    make_plugin_mlm(
                        "answerdotai/ModernBERT-base", window=16, device=device
                    ),
                    make_plugin_case(),
                    make_plugin_propagate(),
                ],
            ),
        ),
    ]

    _run.info[f"ocr_scramble.errors_unit"] = "(WER,CER)"

    def align_setup_test(
        job_i: int, document: Document, strategy: Strategy, wer_cer: tuple[float, float]
    ) -> tuple[int, list[str], float]:
        t0 = time.process_time()
        hashed_chapters = [
            hash_tokens(chapter, hash_len=hash_len) for chapter in document.chapters
        ]
        user_chapters = [
            ocr_scramble(chapter, *wer_cer) for chapter in document.chapters
        ]
        aligned_tokens = strategy.align_fn(hashed_chapters, user_chapters, hash_len)
        t1 = time.process_time()
        return job_i, aligned_tokens, t1 - t0

    setups = list(it.product(corpus, strategies, zip(wer_grid, cer_grid)))
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb) as parallel:
        for job_i, aligned_tokens, duration_s in parallel(
            delayed(align_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            document, strategy, (wer, cer) = setups[job_i]
            gold_tokens = list(flatten(document.chapters))
            setup_name = f"b={document.name}.s={strategy.name}.n=ocr_scramble"
            log_alignment_metrics_(
                _run, setup_name, gold_tokens, aligned_tokens, duration_s
            )
            document.log_alignment_task_metrics(_run, setup_name, aligned_tokens)
            progress.update()
