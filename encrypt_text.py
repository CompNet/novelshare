#!/usr/bin/python3
from typing import List
import functools
import sys
import argparse
import os
import json
import hashlib


@functools.lru_cache
def encrypt_token(token: str) -> str:
    h = hashlib.sha256()
    h.update(token.encode("utf-8"))
    return h.hexdigest()


def encrypt_tokens(tokens: List[str]) -> List[str]:
    """Encrypts the textual content of a novel."""
    return [encrypt_token(token) for token in tokens]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--annot_file", type=str, help="Annotation file.")
    parser.add_argument("--output_file", type=str, help="Output file.")
    args = parser.parse_args()

    if args.annot_files:
        with open(os.path.expanduser(args.annot_file)) as f:
            annotations = json.load(f)
    else:
        annotations = json.loads(sys.stdin.read())

    out_annotations = encrypt_tokens(args.annot_file)

    if args.output_file:
        with open(os.path.expanduser(args.output_file), "w") as f:
            json.dump(out_annotations, f, indent=2)
    else:
        print(json.dumps(out_annotations, indent=2))
