#!/usr/bin/python3
from typing import List
import sys
import argparse
import os
import json
import hashlib


def encrypt_tokens(tokens: List[str]) -> List[str]:
    """Encrypts the textual content of a novel"""
    # dictionary of hashes
    hash_dict = {}

    for token in tokens:
        # new token type: compute hash
        if not token in hash_dict:
            h = hashlib.sha256()
            h.update(token.encode("utf-8"))
            # encrypt token type
            hash_dict[token] = h.hexdigest()[:3]

    print("# word types: {}".format(len(hash_dict)))
    print("# hash types: {}".format(len(set(hash_dict.values()))))
    return annotations


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot_file", type=str, help="Annotation file.", required=True
    )
    parser.add_argument("--output_file", type=str, help="Output file.", required=True)
    args = parser.parse_args()

    annot_file = os.path.expanduser(args.annot_file)
    with open(annot_file) as f:
        annotations = json.load(f)

    out_annotations = encrypt_tokens(args.annot_file)

    output_file = os.path.expanduser(args.output_file)
    with open(output_file, "w") as f:
        json.dump(out_annotations, f, indent=2)
