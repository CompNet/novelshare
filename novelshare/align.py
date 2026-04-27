#!/usr/bin/python3
from typing import Callable, Literal, Optional
import sys, os, argparse, difflib
import functools as ft
from collections import Counter
from more_itertools import flatten
from novelshare.conll import dump_conll2002_bio, load_conll2002_bio
from novelshare.hash import hash_token, hash_tokens
from novelshare.utils import strksplit


def load_user_tokens(path: Optional[str], **kwargs) -> list[str]:
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
# difflib SequenceMatcher opcodes, user_tokens, aligned_tokens, hashed_tokens, hash_len
AlignmentPlugin = Callable[
    [list[OpCode], list[str], list[str], list[str], Optional[int]], list[str]
]


def plugin_propagate(
    opcodes: list[OpCode],
    user_tokens: list[str],
    aligned_tokens: list[str],
    hashed_tokens: list[str],
    hash_len: Optional[int],
) -> list[str]:
    """Propagate previous choices to non-aligned tokens

    This alignment plugins tries to align a substituted or deleted
    token if it was already aligned elsewhere in the text.
    """
    # { hash => most_frequent_decoded_token }
    hash_dict = {}

    for tag, i1, i2, _, _ in opcodes:
        # the user did not supply some tokens, or supplied a wrong
        # token. Maybe we did decode some of these tokens before
        # else and we can use them to retrieve this token.
        if tag == "delete" or tag == "replace":
            for i, hashed_token in enumerate(hashed_tokens[i1:i2]):
                # did we already see this hashed_token before? If not,
                # we need to find the most frequent aligned token
                # corresponding to this hash
                if not hashed_token in hash_dict:
                    same_hash_counter = Counter(
                        [
                            token
                            for hsh, token in zip(hashed_tokens, aligned_tokens)
                            if hsh == hashed_token and token != "[UNK]"
                        ]
                    )
                    most_frequent_decoded_token = (
                        max(same_hash_counter, key=same_hash_counter.get)  # type: ignore
                        if len(same_hash_counter) > 0
                        else None
                    )
                    hash_dict[hashed_token] = most_frequent_decoded_token
                # update the memory of most frequent decoded token for
                # the current hash
                most_frequent_decoded_token = hash_dict[hashed_token]

                if not most_frequent_decoded_token is None:
                    aligned_tokens[i1 + i] = most_frequent_decoded_token

    return aligned_tokens


def make_plugin_propagate() -> AlignmentPlugin:
    return plugin_propagate


def plugin_retokenize(
    opcodes: list[OpCode],
    user_tokens: list[str],
    aligned_tokens: list[str],
    hashed_tokens: list[str],
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
        # match the substituted tokens in hashed_tokens
        tokens_to_split = "".join(user_tokens[j1:j2])

        if len(tokens_to_split) > max_token_len:
            continue

        # we compute the number of substituted tokens: this will be
        # our number of splits
        splits_nb = i2 - i1

        if splits_nb > max_splits_nb:
            continue

        for split in strksplit(tokens_to_split, splits_nb):
            hashed_split = hash_tokens(split, hash_len=hash_len)
            if hashed_split == hashed_tokens[i1:i2]:
                aligned_tokens[i1:i2] = split
                break

    return aligned_tokens


def make_plugin_retokenize(max_token_len: int, max_splits_nb: int) -> AlignmentPlugin:
    return ft.partial(
        plugin_retokenize, max_token_len=max_token_len, max_splits_nb=max_splits_nb
    )


def plugin_mlm(
    opcodes: list[OpCode],
    user_tokens: list[str],
    aligned_tokens: list[str],
    hashed_tokens: list[str],
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
                left = aligned_tokens[i1 + i - window : i1 + i]
                right = aligned_tokens[i1 + i + 1 : i1 + i + window]
                X = left + ["[MASK]"] + right
                X = " ".join(X)  # pipeline expects a string pick the
                # probable token whose hashed form match the hashed
                # gold token
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
                    hashed_cand = hash_token(cand, hash_len)
                    if hashed_cand == hashed_tokens[i1 + i]:
                        aligned_tokens[i1 + i] = cand

    return aligned_tokens


def make_plugin_mlm(
    model: str, window: int, device: Literal["auto", "cuda", "cpu"] = "auto"
) -> AlignmentPlugin:
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
    aligned_tokens: list[str],
    hashed_tokens: list[str],
    hash_len: Optional[int],
) -> list[str]:
    """Fix incorrect user token casing."""
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue

        for k, (user_token, hashed_token) in enumerate(
            zip(user_tokens[j1:j2], hashed_tokens[i1:i2])
        ):
            for casing in [str.lower, str.upper, str.capitalize]:
                hashed_user_token = hash_token(casing(user_token), hash_len=hash_len)
                if hashed_user_token == hashed_token:
                    aligned_tokens[i1 + k] = casing(user_token)

    return aligned_tokens


def make_plugin_case() -> AlignmentPlugin:
    return plugin_case


def plugin_cycle(
    opcodes: list[OpCode],
    user_tokens: list[str],
    aligned_tokens: list[str],
    hashed_tokens: list[str],
    hash_len: Optional[int],
    plugins: list[AlignmentPlugin],
    budget: Optional[int] = None,
) -> list[str]:
    plugin_calls_nb = 0
    lowest_errors = float("inf")
    should_restart = True
    while should_restart:
        should_restart = False

        for plugin in plugins:
            aligned_tokens = plugin(
                opcodes, user_tokens, aligned_tokens, hashed_tokens, hash_len
            )
            plugin_calls_nb += 1

            if not budget is None and plugin_calls_nb == budget:
                return aligned_tokens

            errors = sum(
                1 if ref != pred else 0
                for ref, pred in zip(
                    hashed_tokens,
                    hash_tokens(aligned_tokens, hash_len=hash_len),
                )
            )
            if errors < lowest_errors:
                lowest_errors = errors
                should_restart = True
                break

    return aligned_tokens


def make_plugin_cycle(
    plugins: list[AlignmentPlugin], budget: Optional[int] = None
) -> AlignmentPlugin:
    return ft.partial(plugin_cycle, plugins=plugins, budget=budget)


def _get_opcodes(
    hashed_tokens: list[str] | list[list[str]],
    hashed_user_tokens: list[str] | list[list[str]],
) -> list[OpCode]:
    if isinstance(hashed_tokens[0], str):
        matcher = difflib.SequenceMatcher(None, hashed_tokens, hashed_user_tokens)
        return matcher.get_opcodes()

    assert len(hashed_tokens) == len(hashed_user_tokens)
    cur_i = 0
    cur_j = 0
    opcodes = []
    for block, user_block in zip(hashed_tokens, hashed_user_tokens):
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


def align_tokens(
    hashed_tokens: list[str] | list[list[str]],
    user_tokens: list[str] | list[list[str]],
    hash_len: int | None = None,
    alignment_plugins: list[AlignmentPlugin] | None = None,
) -> list[str]:
    """Attempt to align tokens with annotations using the provided
    user tokens.

    .. note::

        The parameters hashed_tokens, tags and user_tokens can either
        be a list or a list of list.  Using a list of list is useful for
        performance: in that case, the alignment will be computed for
        pairs of smaller sequences, improving performance due to the
        complexity of the alignment algorithm.  This should only be used
        if the input text can be cut in blocks where we can be certain
        that there is no alignment between a token from a block and a
        token from another (for example, chapters from a novel).

    :param hashed_tokens: tokens hashed with SHA-256
    :param tags: NER tags
    :param user_tokens: user tokens, in clear
    :param hash_len: length of the SHA-256 hash (default: 64)
    :param alignment_plugins: a list of alignment plugins to improve
        performance
    """
    if len(hashed_tokens) == 0:
        return []

    is_block_input = isinstance(user_tokens[0], list)

    if is_block_input:
        hashed_user_tokens = [
            hash_tokens(tokens, hash_len=hash_len) for tokens in user_tokens
        ]
    else:
        hashed_user_tokens = hash_tokens(user_tokens, hash_len=hash_len)

    opcodes = _get_opcodes(hashed_tokens, hashed_user_tokens)
    if is_block_input:
        hashed_tokens = list(flatten(hashed_tokens))
        user_tokens = list(flatten(user_tokens))
    aligned_tokens = ["[UNK]" for _ in hashed_tokens]
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            aligned_tokens[i1:i2] = user_tokens[j1:j2]

    if not alignment_plugins is None:
        for plugin in alignment_plugins:
            # the previous plugins may have fixed some errors. We
            # remove the fixed cases there to prevent the next plugins
            # to fix these errors again.
            opcodes = [
                (tag, i1, i2, j1, j2)
                for tag, i1, i2, j1, j2 in opcodes
                if any(t == "[UNK]" for t in aligned_tokens[i1:i2])
            ]
            aligned_tokens = plugin(
                opcodes, user_tokens, aligned_tokens, hashed_tokens, hash_len
            )

    assert len(aligned_tokens) == len(hashed_tokens)
    return aligned_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Input CoNLL-2002 file, with tokens hashed.",
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

    hashed_tokens, tags = load_conll2002_bio(args.input_file, separator=args.separator)
    user_tokens = load_user_tokens(args.user_file)
    aligned_tokens = align_tokens(hashed_tokens, user_tokens)
    dump_conll2002_bio(aligned_tokens, tags, args.output_file, args.separator)
