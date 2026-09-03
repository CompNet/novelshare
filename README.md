# novelshare

Are you looking to reproduce our ACL 2026 article experiments? See the [corresponding branch](https://github.com/CompNet/novelshare/tree/acl2026).

Novelshare is a library that allows to share annotations of a copyrighted corpus, provided the user of the corpus has a (possibly slightly different version) of the copyrighted data.


# Installation

Novelshare is not (yet) on PyPi, but you can install it directly from GitHub with `pip install 'git+https://github.com/CompNet/novelshare'`.


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

## Huggingface `datasets` Integration

For convenience, it is possible to directly hash the column of a huggingface dataset:

```python
from datasets import load_dataset
from novelshare.huggingface import hash_hf_dataset_tokens

my_dataset = load_dataset("compnet-renard/7-romans-ner", split="train") # example dataset

# by default, the library guesses that the column 'tokens' contains
# the tokens of the dataset. It is possible to override this behaviour
# by passing the optional 'tokens_col' parameter.
hashed_dataset = hash_hf_dataset_tokens(my_dataset, hash_len=2)
print(hashed_dataset[0]["tokens"])
# ['12', 'b7', '40', 'b6', 'bc', '66', 'b7', 'de', '95', '4c', '33', '95', 'c3', 'aa', 'd0', 'b1', '8a', '38', '60', 'ee', 'cd']
```

It is also possible to align the of a hashed huggingface dataset with your user tokens:

```python
user_tokens = list(my_dataset["tokens"])
# as above, one can pass the 'tokens_col' parameter.
aligned = align_hf_dataset_tokens(hashed_dataset, user_tokens, hash_len=2)
print(aligned[0]["tokens"])
# ['Quand', 'la', 'caissière', 'lui', 'eut', 'rendu', 'la', 'monnaie', 'de', 'sa', 'pièce', 'de', 'cent', 'sous', ',', 'Georges', 'Duroy', 'sortit', 'du', 'restaurant', '.']
```


# Development setup

## uv (preferred)

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


## guix

We provide a reproducible environment with `guix` (currently with the cpu version of torch only):

```sh
guix time-machine -C channels.scm -- shell -C -m manifest.scm
```


## Tests

We use pytest for testing, so you can run tests with `python -m pytest tests`.



# Citation

If you use novelshare in your research, please cite:

```bibtex
@InProceedings{Amalvy2026,
    author = {Amalvy, A. and Labatut, V. and Bost, X. and Huang, H.-H.},
    title = {Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing},
    year = 2026,
    booktitle = {64th Annual Meeting of the Association for Computational Linguistics (to appear)}, 
}
```
