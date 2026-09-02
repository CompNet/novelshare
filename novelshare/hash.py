#!/usr/bin/python3
from typing import List, Optional
import functools
import argparse
import hashlib
from novelshare.conll import dump_conll2002_bio, load_conll2002_bio


@functools.lru_cache
def hash_token(token: str, hash_len: int | None = None) -> str:
    h = hashlib.sha256()
    h.update(token.encode("utf-8"))
    string = format(int.from_bytes(h.digest(), "big"), "0256b")[2:]
    if not hash_len is None:
        return string[:hash_len]
    return string


def hash_tokens(tokens: list[str], hash_len: int | None = None) -> list[str]:
    return [hash_token(token, hash_len) for token in tokens]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, help="Input CoNLL-2002 file.")
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        default=" ",
        help="Separator between tokens and BIO tags.",
    )
    parser.add_argument("-o", "--output-file", type=str, help="Output CoNLL-2002 file.")
    args = parser.parse_args()

    tokens, tags = load_conll2002_bio(args.input_file, separator=args.separator)
    hashed_tokens = hash_tokens(tokens)
    dump_conll2002_bio(hashed_tokens, tags, args.output_file, args.separator)
