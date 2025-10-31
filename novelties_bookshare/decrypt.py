#!/usr/bin/python3
from typing import Callable, List, Literal, Optional
import sys, os, argparse, difflib
import functools as ft
from more_itertools import flatten
from novelties_bookshare.conll import dump_conll2002_bio, load_conll2002_bio
from novelties_bookshare.encrypt import encrypt_token, encrypt_tokens
from novelties_bookshare.utils import strksplit


def load_user_tokens(path: Optional[str], **kwargs) -> List[str]:
    if not path is None:
        with open(os.path.expanduser(path), **kwargs) as f:
            user_data = f.read()
    else:
        user_data = sys.stdin.read()

    user_tokens = []
    for line in user_data.split("\n"):
        user_tokens.append(line)

    return user_tokens


OpCode = tuple[Literal["replace", "delete", "insert", "equal"], int, int, int, int]
# difflib SequenceMatcher opcodes, user_tokens, decrypted_tokens, encrypted_tokens, hash_len
DecryptPlugin = Callable[
    [list[OpCode], list[str], list[str], list[str], Optional[int]], list[str]
]


def plugin_propagate(
    opcodes: list[OpCode],
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
) -> List[str]:
    """Propagate previous choices to non-decrypted tokens

    This decryption plugins tries to decrypt a substituted or deleted
    token if it was already decryped elsewhere in the text.
    """
    for tag, i1, i2, _, _ in opcodes:
        if tag == "delete" or tag == "replace":
            # the user did not supply some tokens, or supplied a wrong
            # token. Maybe we did decode some of these tokens before
            # else and we can use them to retrieve this token.
            for i, encrypted_token in enumerate(encrypted_tokens[i1:i2]):
                for h, token in zip(encrypted_tokens, decrypted_tokens):
                    if h == encrypted_token:
                        decrypted_tokens[i1 + i] = token
    return decrypted_tokens


def make_plugin_propagate() -> DecryptPlugin:
    return plugin_propagate


def plugin_split(
    opcodes: list[OpCode],
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
    max_token_len: int,
    max_splits_nb: int,
) -> list[str]:
    """Fix incorrect user token merging.

    In the case of a tokenization error, a word can be incorrectly
    merged on the side of the user.  For example:

    .. example::

        ref  user
        ---  ----
        e1   e1
        e2   e2-e3 < substitution
        e3   -
        e4   e4

    In that case, we have a substitution.  We can try all possible
    splits of the merged tokens.  This also works in the reverse case:

    .. example::

        ref  user
        ---  ----
        e1    e1
        e2-e3 e2 < substitution
        -     e3
        e4    e4

    """
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue

        # we will try different splits of the tokens to see if they
        # match the substituted tokens in encrypted_tokens
        tokens_to_split = "".join(user_tokens[j1:j2])

        if len(tokens_to_split) > max_token_len:
            continue

        # we compute the number of substituted tokens: this will be
        # our number of splits
        splits_nb = i2 - i1

        if splits_nb > max_splits_nb:
            continue

        for split in strksplit(tokens_to_split, splits_nb):
            encrypted_split = encrypt_tokens(split, hash_len=hash_len)
            if encrypted_split == encrypted_tokens[i1:i2]:
                decrypted_tokens[i1:i2] = split
                break

    return decrypted_tokens


def make_plugin_split(max_token_len: int, max_splits_nb: int) -> DecryptPlugin:
    return ft.partial(
        plugin_split, max_token_len=max_token_len, max_splits_nb=max_splits_nb
    )


def plugin_mlm(
    opcodes: list[OpCode],
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
    pipeline,
    window: int,
) -> list[str]:
    """
    ref  user
    ---  ----
    e1    e1
    e2    - < deletion
    e3    e3
    e4    e4
    """
    for tag, i1, i2, _, _ in opcodes:
        if tag == "replace" or tag == "delete":
            # the user did not supply some tokens, or supplied a wrong
            # token. In that case, we try to decode the token using BERT
            for i in range(i2 - i1):
                left = decrypted_tokens[i1 + i - window : i1 + i]
                right = decrypted_tokens[i1 + i + 1 : i1 + i + window]
                X = left + ["[MASK]"] + right
                X = " ".join(X)  # pipeline expects a string pick the
                # probable token whose encrypted form match the
                # encrypted gold token
                candidates = pipeline(X)
                # it's possible (although unlikely) that other mask
                # tokens are here. In that case, the pipeline returns
                # a list of candidate list, so we deal with that here
                if "[MASK]" in left or "[MASK]" in right:
                    candidates_index = sum(
                        1 if ltok == "[MASK]" else 0 for ltok in left
                    )
                    candidates = candidates[candidates_index]
                # perform the replacement
                for cand in candidates:
                    cand = cand["token_str"].strip(" ")
                    encrypted_cand = encrypt_token(cand, hash_len)
                    if encrypted_cand == encrypted_tokens[i1 + i]:
                        decrypted_tokens[i1 + i] = cand

    return decrypted_tokens


def make_plugin_mlm(
    model: str, window: int, device: Literal["auto", "cuda", "cpu"] = "auto"
) -> DecryptPlugin:
    from transformers import pipeline
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ft.partial(
        plugin_mlm,
        pipeline=pipeline("fill-mask", model=model, device=torch.device(device)),
        window=window,
    )


def plugin_case(
    opcodes: list[OpCode],
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
) -> list[str]:
    """Fix incorrect user token casing."""
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue

        for k, (user_token, encrypted_token) in enumerate(
            zip(user_tokens[j1:j2], encrypted_tokens[i1:i2])
        ):
            for casing in [str.lower, str.upper, str.capitalize]:
                encrypted_user_token = encrypt_token(
                    casing(user_token), hash_len=hash_len
                )
                if encrypted_user_token == encrypted_token:
                    decrypted_tokens[i1 + k] = user_token

    return decrypted_tokens


def make_plugin_case() -> DecryptPlugin:
    return plugin_case


def plugin_cycle(
    opcodes: list[OpCode],
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
    plugins: list[DecryptPlugin],
    budget: Optional[int] = None,
) -> list[str]:
    plugin_calls_nb = 0
    lowest_errors = float("inf")
    should_restart = True
    while should_restart:
        should_restart = False

        for plugin in plugins:
            decrypted_tokens = plugin(
                opcodes, user_tokens, decrypted_tokens, encrypted_tokens, hash_len
            )
            plugin_calls_nb += 1

            if not budget is None and plugin_calls_nb == budget:
                return decrypted_tokens

            errors = sum(
                1 if ref != pred else 0
                for ref, pred in zip(
                    encrypted_tokens,
                    encrypt_tokens(decrypted_tokens, hash_len=hash_len),
                )
            )
            if errors < lowest_errors:
                lowest_errors = errors
                should_restart = True
                break

    return decrypted_tokens


def make_plugin_cycle(
    plugins: list[DecryptPlugin], budget: Optional[int] = None
) -> DecryptPlugin:
    return ft.partial(plugin_cycle, plugins=plugins, budget=budget)


def _get_opcodes(
    encrypted_tokens: list[str] | list[list[str]],
    encrypted_user_tokens: list[str] | list[list[str]],
) -> list[OpCode]:
    if isinstance(encrypted_tokens[0], str):
        matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
        return matcher.get_opcodes()

    assert len(encrypted_tokens) == len(encrypted_user_tokens)
    cur_i = 0
    cur_j = 0
    opcodes = []
    for block, user_block in zip(encrypted_tokens, encrypted_user_tokens):
        matcher = difflib.SequenceMatcher(None, block, user_block)
        local_opcodes = matcher.get_opcodes()
        global_opcodes = [
            (tag, i1 + cur_i, i2 + cur_i, j1 + cur_j, j2 + cur_j)
            for tag, i1, i2, j1, j2 in local_opcodes
        ]
        opcodes += global_opcodes
        cur_i += len(block)
        cur_j += len(user_block)
    return opcodes


def decrypt_tokens(
    encrypted_tokens: list[str] | list[list[str]],
    user_tokens: list[str] | list[list[str]],
    hash_len: int | None = None,
    decryption_plugins: list[DecryptPlugin] | None = None,
) -> list[str]:
    """Attempt to decrypt tokens using the provided user tokens.

    .. note::

        The parameters encrypted_tokens, tags and user_tokens can either
        be a list or a list of list.  Using a list of list is useful for
        performance: in that case, the alignment will be computed for
        pairs of smaller sequences, improving performance due to the
        complexity of the alignment algorithm.  This should only be used
        if the input text can be cut in blocks where we can be certain
        that there is no alignment between a token from a block and a
        token from another (for example, chapters from a novel).

    :param encrypted_tokens: tokens encrypted with SHA-256
    :param tags: NER tags
    :param user_tokens: user tokens, in clear
    :param hash_len: length of the SHA-256 hash (default: 64)
    :param decryption_plugins: a list of decryption plugins to improve
        performance
    """
    if len(encrypted_tokens) == 0:
        return []

    is_block_input = isinstance(user_tokens[0], list)

    if is_block_input:
        encrypted_user_tokens = [
            encrypt_tokens(tokens, hash_len=hash_len) for tokens in user_tokens
        ]
    else:
        encrypted_user_tokens = encrypt_tokens(user_tokens, hash_len=hash_len)

    opcodes = _get_opcodes(encrypted_tokens, encrypted_user_tokens)
    if is_block_input:
        encrypted_tokens = list(flatten(encrypted_tokens))
        user_tokens = list(flatten(user_tokens))
    decrypted_tokens = ["[UNK]" for _ in encrypted_tokens]
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            decrypted_tokens[i1:i2] = user_tokens[j1:j2]

    if not decryption_plugins is None:
        for plugin in decryption_plugins:
            decrypted_tokens = plugin(
                opcodes, user_tokens, decrypted_tokens, encrypted_tokens, hash_len
            )

    assert len(decrypted_tokens) == len(encrypted_tokens)
    return decrypted_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--encrypted-file", type=str, help="Encrypted CoNLL-2002 file."
    )
    parser.add_argument(
        "-u", "--user-file", type=str, help="Input user file (one token per line)."
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        default=" ",
        help="Separator between tokens and BIO tags.",
    )
    parser.add_argument("-o", "--output-file", type=str, help="Output CoNLL-2002 file.")
    args = parser.parse_args()

    encrypted_tokens, tags = load_conll2002_bio(
        args.encrypted_file, separator=args.separator
    )
    user_tokens = load_user_tokens(args.user_file)
    decrypted_tokens = decrypt_tokens(encrypted_tokens, tags, user_tokens)
    dump_conll2002_bio(decrypted_tokens, tags, args.output_file, args.separator)
