#!/usr/bin/python3
from typing import List, Optional
import sys, os
import argparse
import difflib
from novelties_bookshare.conll import dump_conll2002_bio, load_conll2002_bio
from novelties_bookshare.encrypt import encrypt_tokens


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


def decrypt_tokens(
    encrypted_tokens: list, tags: List[str], user_tokens: List[str]
) -> List[str]:
    assert len(encrypted_tokens) == len(tags)

    decrypted_tokens = ["[UNK]" for _ in encrypted_tokens]

    encrypted_user_tokens = encrypt_tokens(user_tokens)

    # loop over operations turning encrypted_tokens into
    # encrypted_user_tokens
    matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_user_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(tag, i1, i2, j1, j2)
        # NOTE: ignores 'insert', 'replace', 'delete'
        if tag == "equal":
            decrypted_tokens[i1:i2] = user_tokens[j1:j2]
        print(decrypted_tokens)

    # second pass
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete" or tag == "replace":
            # the user did not supply some tokens, or supplied a wrong
            # token. Maybe we did decode some of these tokens before
            # else and we can use them to retrieve this token.
            for i, encrypted_token in enumerate(encrypted_tokens[i1:i2]):
                for h, token in zip(encrypted_tokens, decrypted_tokens):
                    if h == encrypted_token:
                        decrypted_tokens[i1 + i] = token

    assert len(decrypted_tokens) == len(tags)
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
