#!/usr/bin/python3
from typing import List
import sys
import argparse
import difflib
from nltk.tokenize import word_tokenize
from encrypt_text import encrypt_tokens


def decrypt_tokens(
    encrypted_tokens: list, tags: List[str], noisy_tokens: List[str]
) -> List[str]:
    assert len(encrypted_tokens) == len(tags)

    decrypted_tokens = ["[UNK]" for _ in encrypted_tokens]

    encrypted_noisy_tokens = encrypt_tokens(noisy_tokens)

    # loop over operations turning encrypted_tokens into
    # encrypted_noisy_tokens
    matcher = difflib.SequenceMatcher(None, encrypted_tokens, encrypted_noisy_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(tag, i1, i2, j1, j2)
        # NOTE: ignores 'insert', 'replace', 'delete'
        if tag == "equal":
            decrypted_tokens[i1:i2] = noisy_tokens[j1:j2]
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

    return decrypted_tokens


ref_tokens = "A B C D E E".split()
tags = ["B-PER", "O", "O", "O", "B-PER", "I-PER"]
noisy_tokens = "A B C D E X".split()
print(f"{ref_tokens=}")
print(f"{tags=}")
print(f"{noisy_tokens=}")
tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, noisy_tokens)
print(f"{tokens=}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--annot_file", type=str, help="Annotation file.", required=True
#     )

#     parser.add_argument(
#         "--subtitles_dir",
#         type=str,
#         help="Directory containing subtitles.",
#         required=True,
#     )

#     parser.add_argument(
#         "--subtitles_encoding", type=str, help="Subtitles encoding.", default="utf-8"
#     )

#     parser.add_argument(
#         "--output_annot_file", type=str, help="Output annotation file.", required=True
#     )
#     args = parser.parse_args()

#     # TODO
