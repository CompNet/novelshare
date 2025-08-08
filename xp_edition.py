import pathlib as pl
from collections import defaultdict
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
    make_plugin_case,
    make_plugin_cycle,
)
from novelties_bookshare.experiments.data import load_book
from tqdm import tqdm

ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))

EDITION_SETS = {
    "Brave_New_World": {
        "HC98": "./data/editions_diff/Brave_New_World/HC98",
        "HC06": "./data/editions_diff/Brave_New_World/HC06",
        "HC04": "./data/editions_diff/Brave_New_World/HC04",
        "RB06": "./data/editions_diff/Brave_New_World/RB06",
    }
}


@ex.config
def config():
    novelties_path: str
    edition_set: str


@ex.automain
def main(_run: Run, novelties_path: str, edition_set: str):
    print_config(_run)
    assert edition_set in EDITION_SETS

    novelties_tokens, novelties_tags = load_book(
        pl.Path(novelties_path).expanduser() / "corpus" / edition_set
    )

    wild_editions = {
        key: load_book(path)[0] for key, path in EDITION_SETS[edition_set].items()
    }

    def normalize_(tokens: list[str], replacements: list[tuple[list[str], str]]):
        for i, token in enumerate(tokens):
            for repl_source, repl_target in replacements:
                if token in repl_source:
                    tokens[i] = repl_target

    # OPTIONAL: preprocessing
    normalize_(novelties_tokens, [(["``", "''"], '"')])
    normalize_(novelties_tokens, [(["…"], "...")])
    for tokens in wild_editions.values():
        normalize_(tokens, [(["``", "''", "“", "”"], '"')])
        normalize_(tokens, [(["‘", "’"], "'")])
        normalize_(tokens, [(["…"], "...")])
        normalize_(tokens, [(["—"], "-")])

    novelties_encrypted_tokens = encrypt_tokens(novelties_tokens)

    strategies = {
        "naive": None,
        "case": [make_plugin_case()],
        "propagate": [make_plugin_propagate()],
        "splice": [make_plugin_split(max_token_len=24, max_splits_nb=4)],
        "bert": [make_plugin_mlm("answerdotai/ModernBERT-base", window=16)],
        "pipe": [
            make_plugin_propagate(),
            make_plugin_case(),
            make_plugin_split(max_token_len=24, max_splits_nb=4),
            make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
            make_plugin_propagate(),
        ],
        "cycle": [
            make_plugin_cycle(
                [
                    make_plugin_propagate(),
                    make_plugin_case(),
                    make_plugin_split(max_token_len=24, max_splits_nb=4),
                    make_plugin_mlm("answerdotai/ModernBERT-base", window=16),
                ],
                budget=None,
            )
        ],
    }
    # { strategy => { edition => { token => number of error } } }
    errors = {
        strat: {ed: defaultdict(int) for ed in wild_editions.keys()}
        for strat in strategies.keys()
    }

    progress = tqdm(total=len(wild_editions) * len(strategies), ascii=True)

    for edition, user_tokens in wild_editions.items():
        for strat, strat_plugins in strategies.items():
            progress.set_description(f"{edition}.{strat}")

            decrypted_tokens = decrypt_tokens(
                novelties_encrypted_tokens,
                novelties_tags,
                user_tokens,
                hash_len=None,
                decryption_plugins=strat_plugins,
            )
            local_errors_nb = sum(
                1 if ref != pred else 0
                for ref, pred in zip(novelties_tokens, decrypted_tokens)
            )
            setup_name = f"s={strat}.e={edition}"
            _run.log_scalar(f"{setup_name}.errors_nb", local_errors_nb)

            for ref, pred in zip(novelties_tokens, decrypted_tokens):
                if ref != pred:
                    errors[strat][edition][ref] += 1

            progress.update()

    _run.info["errors"] = errors
