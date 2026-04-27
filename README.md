# novelshare


# Installation

## uv

Clone the repository and use `uv`:

```sh
uv sync
```

By default this install everything for and the cuda version of PyTorch. We have several extras to configure your installation. For development dependencies:

```sh
uv sync --dev
```

For GPU acceleration, depending on your GPU, you can use:

```sh
# if you have a cuda 12.8 compatible GPU
uv sync --extra cu128
# if you have a rocm 6.3 compatible GPU
uv sync --extra rocm63
```

The `cpu` extra also exists for the cpu version of torch.

To not only use the library, but also add additional dependencies to reproduce experiments, you can use the `experiments` extra:

```sh
uv sync --extra experiments
```


## guix

We provide a more reproducible environment using `guix`:

```sh
guix time-machine -C channels.scm -- shell -C -m manifest.scm
```


# Reproducing Experiments

After installation, activate your Python environment (`source .venv/bin/activate`). You can then reproduce experiments by launching these scripts: 

| Section                                 | Experiment Collection Script       |
|-----------------------------------------|------------------------------------|
| 4.4 Results Per Edition                 | `./xp_edition.sh`                  |
| 4.5 Synthetic Errors                    | `./xp_edition_synthetic_errors.sh` |
| Appendix E/F                            | `./xp_ner.sh`                      |
|-----------------------------------------|------------------------------------|
| All experiments (can take a long time!) | `./all_xp.sh`                      |

Each shell script launches a collection of experiments (corresponding to launching xp_*.py scripts with different parameters). Each experiment creates a directory under `./runs`.


## Plots

After running experiments, you can use the plotting scripts to reproduce the figures from the paper:

| Figure    | Experiment Script                | Plot Command                                                                                                                                                                                                                                                                                                                                                 |
|-----------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Figure 4  | -                                | `python plot_hash_collisions.py -l`                                                                                                                                                                                                                                                                                                                          |
| Figure 5  | `xp_edition.sh`                  | `python plot_xp_edition_hash_len.py -r ./runs/xp_edition_n=Frankenstein_h=* ./runs/xp_edition_n=Moby_Dick_h=* ./runs/xp_edition_n=Pride_and_Prejudice_h=* -m errors_percent -l`                                                                                                                                                                              |
| Figure 6  | `xp_edition.sh`                  | `python plot_xp_edition.py -r ./runs/xp_edition_n=Frankenstein_h=2 ./runs/xp_edition_n=Moby_Dick_h=2 ./runs/xp_edition_n=Pride_and_Prejudice_h=2 -m errors_percent`                                                                                                                                                                                          |
| Figure 7  | `xp_edition_synthetic_errors.sh` | `python plot_xp_synthetic_errors.py -m errors_percent -r ./runs/xp_synthetic_errors_h=2/ -c ./runs/xp_synthetic_errors_ocr_h=2/ -o ./plots/synthetic_errors`                                                                                                                                                                                                 |
| Figure 8  | `xp_edition.sh`                  | `python plot_xp_edition.py -r ./runs/xp_edition_n=Frankenstein_h=2 ./runs/xp_edition_n=Moby_Dick_h=2 ./runs/xp_edition_n=Pride_and_Prejudice_h=2 -m duration_s -l`                                                                                                                                                                                           |
| Figure 9  | `xp_ner.sh`                      | `python plot_xp_synthetic_errors.py -r ./runs/xp_synthetic_errors_c=conll2003_h=2/ -c ./runs/xp_synthetic_errors_ocr_c=conll2003_h=2/ -m errors_percent -o ./plots/synthetic_errors_conll2003`                                                                                                                                                               |
| Figure 10 | `xp_ner.sh`                      | `python plot_xp_synthetic_errors.py -r ./runs/xp_synthetic_errors_c=wnut2017_h=2/ -c ./runs/xp_synthetic_errors_ocr_c=wnut2017_h=2/ -m errors_percent -o ./plots/synthetic_errors_wnut2017`                                                                                                                                                                  |
| Figure 11 | `xp_ner.sh`                      | `python plot_task_specific_errors.py -r ./runs/xp_synthetic_errors_c=conll2003_h=2/ ./runs/xp_synthetic_errors_c=wnut2017_h=2/ -c ./runs/xp_synthetic_errors_ocr_c=conll2003_h=2/ ./runs/xp_synthetic_errors_ocr_c=wnut2017_h=2/ -t entity_errors_percent_strict entity_errors_percent_strict -m errors_percent errors_percent  -l 'CoNLL 2003' 'WNUT 2017'` |
| Figure 12 | `xp_edition.sh`                  | `python plot_xp_edition.py -r ./runs/xp_edition_mlm_params_n=* -m errors_percent`                                                                                                                                                                                                                                                                            |
| Figure 13 | `xp_edition.sh`                  | `python plot_xp_edition.py -r ./runs/xp_edition_n=*_h=2 -m precision_errors_nb -l`                                                                                                                                                                                                                                                                           |

For all plotting scripts, you can use `--help` for more details. 


# Library user guide

## Hashing your corpus

```python
from novelshare.hash import hash_tokens

# assuming my_tokens is a list of tokens, and my_annotations is a list
# of single or multiple annotations (one or more annotations per
# token)
my_tokens, my_annotations = load_my_corpus()

# hash tokens with the desired hash length (2 is a solid default
# value)
hashed_tokens = hash_tokens(my_tokens, hash_len=2)

with open("hashed_corpus.conll", "w") as f:
    for token, annotations in zip(hashed_tokens, my_annotations):
        f.write(f"{token} {annotations}\n")
```

## Aligned Annotations of a Shared Hashed Corpus

Aligning tokens is done using the `novelshare.align.align_tokens` function:

```python
from novelshare.align import align_tokens

# let's suppose you wish to align your own tokens with a hashed corpus

# 1. load your own tokens
my_tokens = load_my_tokens()

# 2. load the hashed tokens
hashed_tokens = load_hashed_tokens()
annotations = load_annotations()
# each token has one or more annotations
assert len(hashed_tokens) == len(annotations)

# you can align your tokens to annotations using align_tokens!
aligned_tokens = align_tokens(hashed_tokens, my_tokens, hash_len=2)
assert len(aligned_tokens) == len(annotations)
```

The more the user tokens differ from the source tokens, the more errors will occur in the alignment process. It is possible to use additional alignment plugins to improve performance. Here are some examples:

```python
from novelshare.align import (
    make_plugin_propagate,
    make_plugin_mlm,
    make_plugin_retokenize,
    make_plugin_case,
)

# Option #1: ligthweight but effective, using the propagate plugin alone
aligned = align_tokens(
    hashed_tokens,
    my_tokens,
    hash_len=2,
    alignnment_plugins=[make_plugin_propagate()],
)

# Option #2: heavier but more powerful, using a sequence of plugins
aligned = align_tokens(
    hashed_tokens,
    my_tokens,
    hash_len=2,
    alignment_plugins=[
        make_plugin_propagate(),
        make_plugin_case(),
        make_plugin_retokenize(max_token_len=16, max_splits_nb=8),
    ],
)

# Option #2: heaviest but the most powerful, using masked language
# modeling to end the sequence of plugins
aligned = align_tokens(
    hashed_tokens,
    my_tokens,
    hash_len=2,
    alignment_plugins=[
        make_plugin_propagate(),
        make_plugin_case(),
        make_plugin_retokenize(max_token_len=16, max_splits_nb=8),
        # if you have a GPU, you can pass device="cuda" for GPU
        # accelerated inference.
        make_plugin_mlm("answerdotai/ModernBERT-base", window=32)
    ],
)
```

Adding alignment plugins is, however, also increasing runtime. To reduce runtime, it is possible to take advantage of the fact that a dataset might be *chunked*. This can happen, for example, in the case of a book divided into chapters. Since `novelshare` uses `difflib` to align sequences which is O(n^2), it is usually noticeably faster to align chapters separately rather than aligning the whole document at once. The drawback is that one needs aligned chapters. `novelshare` support this usecase out of the box, as the `align_tokens` function can take a list of tokens or a list of chunks:

```python
from typing import Any
from novelshare.align import align_tokens

hashed_chapters: list[list[str]] = load_chapters()
annotations: list[list[Any]] = load_annotations()

my_chapters: list[list[str]] = load_my_chapters()

# align_tokens supports list of chunks out of the box!
aligned = align_tokens(hashed_chapters, my_chapters, hash_len=2)
```
