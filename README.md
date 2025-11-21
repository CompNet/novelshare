# novelties-bookshare


# Library user guide

## Encrypting your corpus

```python
from novelties_bookshare.encrypt import encrypt_tokens

# assuming my_tokens is a list of tokens, and my_annotations is a list
# of single or multiple annotations (one or more annotations per
# token)
my_tokens, my_annotations = load_my_corpus()

# encrypt tokens with the desired hash length (2 is a solid default
# value)
encrypted_tokens = encrypt_tokens(my_tokens, hash_len=2)

with open("encrypted_corpus", "w") as f:
    for token, annotations in encrypt_tokens, my_annotations:
        f.write(f"{token} {annotations}\n")
```

## Decrypting a shared encrypted corpus

Decrypting tokens is done using the `novelties_bookshare.decrypt.decrypt_tokens` function:

```python
from novelties_bookshare.decrypt import decrypt_tokens

# let's suppose you wish to decrypt an encrypted corpus 
encrypted_source_tokens = load_source_tokens()
source_annotations = load_source_annotations()
# each token has one or more annotations
assert len(source_tokens) == len(source_annotations)

# let's suppose you have access to a degraded version of the tokens of
# the source corpus.
my_tokens = load_my_tokens()

# you can recover the source tokens using decrypt_tokens!
decrypted = decrypt_tokens(encrypted_source_tokens, my_tokens, hash_len=2)
```

The more the user tokens differ from the source tokens, the more errors will occur in the decryption process. It is possible to use additional decryption plugins to improve performance. Here is some examples:

```python
from novelties_bookshare.decrypt import (
    make_plugin_propagate,
    make_pluging_mlm,
    make_plugin_split,
    make_plugin_case,
)

# Option #1: ligthweight but effective, using the propagate plugin alone
decrypted = decrypt_tokens(
    encrypted_source_tokens,
    my_tokens,
    hash_len=2,
    decryption_plugins=[make_plugin_propagate()],
)

# Option #2: heavier but more powerful, using a sequence of plugins
decrypted = decrypt_tokens(
    encrypted_source_tokens,
    my_tokens,
    hash_len=2,
    decryption_plugins=[
        make_plugin_propagate(),
        make_plugin_case(),
        make_plugin_split(max_token_len=16, max_splits_nb=8),
    ],
)

# Option #2: heaviest but the most powerful, using masked language
# modeling to end the sequence of plugin
decrypted = decrypt_tokens(
    encrypted_source_tokens,
    my_tokens,
    hash_len=2,
    decryption_plugins=[
        make_plugin_propagate(),
        make_plugin_case(),
        make_plugin_split(max_token_len=16, max_splits_nb=8),
        make_plugin_mlm("answerdotai/ModernBERT-base", window=32, device=device),
    ],
)
```

Adding plugin is, however, also increasing runtime. To reduce runtime, it is possible to take advantage of the fact that a dataset might be *chunked*. This can happen, for example, in the case of a book divided into chapters. Since `novelties-bookshare` uses `difflib` to align sequences which is O(n^2), it is usually noticeably faster to align chapters separately rather than aligning the whole document at once. The drawback is that one need aligned chapters. `novelties-bookshare` support this usecase out of the box, as the `decrypt_tokens` function can take a list of token or a list of chunks:

```python
from typing import Any
from novelties_bookshare.decrypt import decrypt_tokens

encrypted_source_chapters: list[list[str]] = load_source_chapters()
source_annotations list[list[Any]] = load_source_annotations()

my_chapters: list[list[str]] = load_my_chapters()

# decrypt_tokens supports list of chunks out of the box!
decrypted = decrypt_tokens(encrypted_source_chapters, my_chapters, hash_len=2)
```
