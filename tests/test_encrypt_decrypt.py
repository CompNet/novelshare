from typing import List
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens, plugin_propagate
from hypothesis import given, strategies as st


def test_substitution():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == "A B C D E [UNK]".split()


def test_substitution_propagate():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(
        encrypt_tokens(ref_tokens),
        tags,
        user_tokens,
        decryption_plugins=[plugin_propagate],
    )
    assert pred_tokens == ref_tokens


def test_deletion():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C E E".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == "A B C [UNK] E E".split()


def test_addition():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C X D E E".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == ref_tokens


@given(st.lists(st.text()))
def test_encrypt_decrypt_recover_original_tokens(tokens: List[str]):
    tags = ["O"] * len(tokens)
    assert decrypt_tokens(encrypt_tokens(tokens), tags, tokens) == tokens
