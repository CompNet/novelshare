#!/usr/bin/python3
from typing import Callable, List, Optional
import sys, os, argparse, difflib
import functools as ft
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


# matcher, user_tokens, decrypted_tokens, encrypted_tokens, hash_len
DecryptPlugin = Callable[
    [difflib.SequenceMatcher, list[str], list[str], list[str], Optional[int]], list[str]
]


def decryptplugin_propagate(
    matcher: difflib.SequenceMatcher,
    user_tokens: list[str],
    decrypted_tokens: list[str],
    encrypted_tokens: list[str],
    hash_len: Optional[int],
) -> List[str]:
    """Propagate previous choices to non-decrypted tokens

    This decryption plugins tries to decrypt a substituted or deleted
    token if it was already decryped elsewhere in the text.
    """
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete" or tag == "replace":
            # the user did not supply some tokens, or supplied a wrong
            # token. Maybe we did decode some of these tokens before
            # else and we can use them to retrieve this token.
            for i, encrypted_token in enumerate(encrypted_tokens[i1:i2]):
                for h, token in zip(encrypted_tokens, decrypted_tokens):
                    if h == encrypted_token:
                        decrypted_tokens[i1 + i] = token
    return decrypted_tokens


def make_decryptplugin_propagate() -> DecryptPlugin:
    return decryptplugin_propagate


def decryptplugin_splice(
    matcher: difflib.SequenceMatcher,
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
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
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


def make_decryptplugin_splice(max_token_len: int, max_splits_nb: int) -> DecryptPlugin:
    return ft.partial(
        decryptplugin_splice, max_token_len=max_token_len, max_splits_nb=max_splits_nb
    )


def decryptplugin_mlm(
    matcher: difflib.SequenceMatcher,
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
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" or tag == "delete":
            # the user did not supply some tokens, or supplied a wrong
            # token. In that case, we try to decode the token using BERT
            for i in range(i2 - i1):
                left = decrypted_tokens[i1 + i - window : i1 + i]
                right = decrypted_tokens[i1 + i + 1 : i1 + i + window]
                assert not "[MASK]" in left and not "[MASK]" in right
                X = left + ["[MASK]"] + right
                X = " ".join(X)  # pipeline expects a string
                # pick the probable token whose encrypted form match
                # the encrypted gold token
                candidates: list[dict] = pipeline(X)
                for cand in candidates:
                    cand = cand["token_str"].strip(" ")
                    encrypted_cand = encrypt_token(cand, hash_len)
                    if encrypted_cand == encrypted_tokens[i1 + i]:
                        decrypted_tokens[i1 + i] = cand

    return decrypted_tokens


def make_decryptplugin_mlm(model: str, window: int) -> DecryptPlugin:
    from transformers import pipeline

    return ft.partial(
        decryptplugin_mlm, pipeline=pipeline("fill-mask", model=model), window=window
    )


def decrypt_tokens(
    encrypted_tokens: list,
    tags: List[str],
    user_tokens: List[str],
    hash_len: Optional[int] = None,
    decryption_plugins: Optional[List[DecryptPlugin]] = None,
) -> list[str]:
    assert len(encrypted_tokens) == len(tags)

    decrypted_tokens = ["[UNK]" for _ in encrypted_tokens]

    encrypted_user_tokens = encrypt_tokens(user_tokens, hash_len=hash_len)

    # loop over operations turning encrypted_tokens into
    # encrypted_user_tokens
    matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # NOTE: ignores 'insert', 'replace', 'delete'
        if tag == "equal":
            decrypted_tokens[i1:i2] = user_tokens[j1:j2]

    if not decryption_plugins is None:
        for plugin in decryption_plugins:
            decrypted_tokens = plugin(
                matcher, user_tokens, decrypted_tokens, encrypted_tokens, hash_len
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
