#!/usr/bin/python3
from typing import Callable, List, Optional
import sys, os, argparse, difflib
import functools as ft
from novelties_bookshare.conll import dump_conll2002_bio, load_conll2002_bio
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.utils import iterate_pattern


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


# matcher, decrypted_tokens, encrypted_tokens
DecryptPlugin = Callable[[difflib.SequenceMatcher, List[str], List[str]], List[str]]


def decryptplugin_propagate(
    matcher: difflib.SequenceMatcher,
    decrypted_tokens: List[str],
    encrypted_tokens: List[str],
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


def decryptplugin_splice(
    matcher: difflib.SequenceMatcher,
    decrypted_tokens: List[str],
    encrypted_tokens: List[str],
) -> List[str]:
    """Fix incorrect user token merging.

    In the case of a tokenization error, a word can be incorrectly
    merged on the side of the user.  For example:

    .. example::

        ref user
        --- ----
        e1  e1
        e2  e2-e3 < substitution
        e3  -     < deletion
        e4  e4

    In that case, we have a substitution + a deletion.  We can try all
    possible splits of the merged tokens.
    """
    opcode_tags = [tag for tag, *_ in matcher.get_opcodes()]
    # TODO: if n tokens are merged with n>2, we should have n-1
    # deletions
    # TODO: deletions can appear before substitution
    for starti in iterate_pattern(opcode_tags, ["replace", "delete"]):  # delete*
        ri1, ri2, rj1, rj2 = opcode_tags[starti][1:]
        di1, di2, dj1, dj2 = opcode_tags[starti][1:]
        # TODO:


def decryptplugin_mlm(
    matcher: difflib.SequenceMatcher,
    decrypted_tokens: List[str],
    encrypted_tokens: List[str],
    pipeline,
    window: int,
) -> List[str]:
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" or tag == "delete":
            # the user did not supply some tokens, or supplied a wrong
            # token. In that case, we try to decode the token using BERT
            for i in range(i2 - i1):
                X = decrypted_tokens[i1 + i - window : i1 + i + window]
                assert not "[MASK]" in X
                X[i1 + i] = "[MASK]"  # mask the central token
                X = " ".join(X)  # pipeline expects a string
                # pick the most probable token according to the model
                decrypted_tokens[i1 + i] = pipeline(X)[0]["token_str"]
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
) -> List[str]:
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
            decrypted_tokens = plugin(matcher, decrypted_tokens, encrypted_tokens)

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
