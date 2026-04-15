from typing import Callable, Literal, Optional, Any
import time
import pathlib as pl
import functools as ft
import itertools as it
from dataclasses import dataclass
from datasets import load_dataset as hf_load_dataset
from more_itertools import flatten
from tqdm import tqdm
import numpy as np
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
from novelties_bookshare.experiments.data import iter_book_chapters
from novelties_bookshare.experiments.metrics import (
    errors_nb,
    errors_percent,
    entity_errors_nb,
    entity_errors_percent,
    log_ner_task_metrics_,
)
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


@dataclass
class Document:
    name: str
    chapters: list[list[str]]
    annotations: list[list[Any]] | None = None

    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        pass


CorpusID = Literal["3novels", "conll2003", "wnut2017"]


class Conll2003Document(Document):
    ID2LABEL = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        assert not self.annotations is None
        ref_tokens = list(flatten(self.chapters))
        ref_tags = [self.ID2LABEL[tag_id] for tag_id in flatten(self.annotations)]
        log_ner_task_metrics_(_run, setup_name, ref_tokens, ref_tags, aligned_tokens)


class WNUT2017Document(Document):
    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        assert not self.annotations is None
        ref_tokens = list(flatten(self.chapters))
        ref_tags = list(flatten(self.annotations))
        log_ner_task_metrics_(_run, setup_name, ref_tokens, ref_tags, aligned_tokens)


def load_corpus(name: CorpusID, **kwargs) -> list[Document]:
    if name == "3novels":
        return [
            Document(
                "F-1818",
                list(iter_book_chapters("./data/Frankenstein/F-1818", **kwargs)),
            ),
            Document(
                "MD-1851-US",
                list(iter_book_chapters("./data/Moby_Dick/MD-1851-US", **kwargs)),
            ),
            Document(
                "PP-1813",
                list(
                    iter_book_chapters("./data/Pride_and_Prejudice/PP-1813", **kwargs)
                ),
            ),
        ]
    elif name == "conll2003":
        conll2003 = hf_load_dataset("BramVanroy/conll2003")
        return [
            Conll2003Document(
                split,
                [row["tokens"] for row in conll2003[split]],
                annotations=[row["ner_tags"] for row in conll2003[split]],
            )
            for split in ["train", "validation", "test"]
        ]
    elif name == "wnut2017":
        wnut2017 = hf_load_dataset("extraordinarylab/wnut2017")
        return [
            WNUT2017Document(
                split,
                [row["tokens"] for row in wnut2017[split]],
                annotations=[row["ner_tags"] for row in wnut2017[split]],
            )
            for split in ["train", "validation", "test"]
        ]
    raise ValueError(name)


@ex.config
def config():
    min_error_ratio: float
    max_error_ratio: float
    error_ratio_step: float
    hash_len: int = 64
    corpus_id: CorpusID = "3novels"
    # only used when corpus_id == '3novels'
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
    corpus_id: CorpusID,
    chapter_limit: Optional[int],
    jobs_nb: int,
    device: Literal["auto", "cuda", "cpu"],
):
    print_config(_run)
    assert min_error_ratio >= 0
    assert max_error_ratio > min_error_ratio
    assert hash_len > 0 and hash_len <= 64

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

    errors_fns = [substitute, delete, add, token_split, token_merge]
    for errors_fn in errors_fns:
        _run.info[f"{errors_fn.__name__}.errors_unit"] = "Ratio of syntethic errors"

    error_ratio = [
        float(i) for i in np.arange(min_error_ratio, max_error_ratio, error_ratio_step)
    ]

    def align_setup_test(
        job_i: int,
        document: Document,
        strategy: Strategy,
        errors_fn: Callable[[list[str], int], list[str]],
        error_ratio: float,
    ) -> tuple[int, list[str], float]:
        t0 = time.process_time()
        hashed_chapters = [
            hash_tokens(chapter, hash_len=hash_len) for chapter in document.chapters
        ]
        user_chapters = [
            errors_fn(chapter, int(len(chapter) * error_ratio))
            for chapter in document.chapters
        ]
        aligned_tokens = strategy.align_fn(hashed_chapters, user_chapters, hash_len)
        t1 = time.process_time()
        return job_i, aligned_tokens, t1 - t0

    setups = list(it.product(corpus, strategies, errors_fns, error_ratio))
    progress = tqdm(total=len(setups), ascii=True)

    with Parallel(n_jobs=jobs_nb, return_as="generator_unordered") as parallel:
        for job_i, aligned_tokens, duration_s in parallel(
            delayed(align_setup_test)(i, *args) for i, args in enumerate(setups)
        ):
            document, strategy, errors_fn, error_ratio = setups[job_i]
            ref_tokens = list(flatten(document.chapters))
            setup_name = f"b={document.name}.s={strategy.name}.n={errors_fn.__name__}"
            _run.log_scalar(
                f"{setup_name}.errors_nb",
                errors_nb(ref_tokens, aligned_tokens),
                step=error_ratio,
            )
            _run.log_scalar(
                f"{setup_name}.errors_percent",
                errors_percent(ref_tokens, aligned_tokens),
                step=error_ratio,
            )
            _run.log_scalar(f"{setup_name}.duration_s", duration_s)
            document.log_alignment_task_metrics(_run, setup_name, aligned_tokens)

            progress.update()
