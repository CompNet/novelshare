#!/usr/bin/python3
from typing import List, Optional
import functools
import argparse
import hashlib
from novelties_bookshare.conll import dump_conll2002_bio, load_conll2002_bio


@functools.lru_cache
def encrypt_token(token: str, hash_len: Optional[int] = None) -> str:
    h = hashlib.sha256()
    h.update(token.encode("utf-8"))
    if not hash_len is None:
        return h.hexdigest()[:hash_len]
    return h.hexdigest()


def encrypt_tokens(tokens: List[str], hash_len: Optional[int] = None) -> List[str]:
    return [encrypt_token(token, hash_len) for token in tokens]


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
    encrypted_tokens = encrypt_tokens(tokens)
    dump_conll2002_bio(tokens, tags, args.output_file, args.separator)
