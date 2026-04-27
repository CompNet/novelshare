from typing import Literal, Optional
import time
from more_itertools import flatten
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from novelshare.hash import hash_tokens
from novelshare.align import (
    align_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_retokenize,
    make_plugin_case,
)
from novelshare.experiments.data import (
    iter_book_chapters,
    normalize_,
    EDITION_SETS,
)
from novelshare.experiments.metrics import log_alignment_metrics_, errors

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))


@ex.config
def config():
    novel: str
    hash_len: int = 64
    chapter_limit: Optional[int] = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    retokenize_max_token_len: int = 16
    retokenize_max_splits_nb: int = 8
    mlm_window: int = 32


@ex.automain
def main(
    _run: Run,
    novel: str,
    hash_len: int,
    chapter_limit: Optional[int],
    device: Literal["auto", "cuda", "cpu"],
    retokenize_max_token_len: int,
    retokenize_max_splits_nb: int,
    mlm_window: int,
):
    print_config(_run)
    assert novel in EDITION_SETS
    assert hash_len > 0 and hash_len <= 64

    reference_edition = list(EDITION_SETS[novel].keys())[0]
    reference_chapters = list(
        iter_book_chapters(
            EDITION_SETS[novel][reference_edition], chapter_limit=chapter_limit
        )
    )

    wild_editions = {
        key: list(iter_book_chapters(path, chapter_limit=chapter_limit))
        for key, path in EDITION_SETS[novel].items()
        if key != reference_edition
    }

    normalize_(reference_chapters)
    for chapters in wild_editions.values():
        normalize_(chapters)

    strategies = {
        "naive": None,
        "case": [make_plugin_case()],
        "propagate": [make_plugin_propagate()],
        "retokenize": [
            make_plugin_retokenize(
                max_token_len=retokenize_max_token_len,
                max_splits_nb=retokenize_max_splits_nb,
            )
        ],
        "mlm": [
            make_plugin_mlm(
                "answerdotai/ModernBERT-base", window=mlm_window, device=device
            )
        ],
        "pipe": [
            make_plugin_retokenize(
                max_token_len=retokenize_max_token_len,
                max_splits_nb=retokenize_max_splits_nb,
            ),
            make_plugin_mlm(
                "answerdotai/ModernBERT-base", window=mlm_window, device=device
            ),
            make_plugin_case(),
            make_plugin_propagate(),
        ],
    }
    # { strategy => { edition => { ref_token => [ incorrect pred tokens ] } } }
    all_errors = {
        strat: {ed: {} for ed in wild_editions.keys()} for strat in strategies.keys()
    }

    progress = tqdm(total=len(wild_editions) * len(strategies), ascii=True)

    for edition, user_tokens in wild_editions.items():
        reference_hashed = [
            hash_tokens(chapter, hash_len=hash_len) for chapter in reference_chapters
        ]
        # If we have the same number of chapters, we assume that
        # the chapters are aligned. Otherwise, we have to perform
        # alignment on the entire novels, not on individual
        # chapters (more costly!)
        same_number_of_chapters = len(reference_chapters) == len(user_tokens)
        if not same_number_of_chapters:
            user_tokens = list(flatten(user_tokens))
            reference_hashed = list(flatten(reference_hashed))

        for strat, strat_plugins in strategies.items():
            progress.set_description(f"{edition}.{strat}")

            t0 = time.process_time()
            aligned_tokens = align_tokens(
                reference_hashed,
                user_tokens,
                hash_len=hash_len,
                alignment_plugins=strat_plugins,
            )
            t1 = time.process_time()

            reference_tokens = list(flatten(reference_chapters))
            setup_name = f"s={strat}.e={novel},{edition}"
            log_alignment_metrics_(
                _run,
                setup_name,
                reference_tokens,
                aligned_tokens,
                t1 - t0,
            )
            all_errors[strat][edition] = errors(reference_tokens, aligned_tokens)

            progress.update()

    _run.info["errors"] = all_errors
