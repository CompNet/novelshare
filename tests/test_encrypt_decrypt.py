from hypothesis import given, strategies as st
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import (
    decrypt_tokens,
    make_plugin_case,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_split,
)
from novelties_bookshare.experiments.metrics import errors_nb
from tests.strategies import error_seq_pairs


def test_substitution():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), user_tokens)
    assert pred_tokens == "A B C D E [UNK]".split()


def test_substitution_propagate():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    pred_tokens = decrypt_tokens(
        encrypt_tokens(ref_tokens),
        user_tokens,
        decryption_plugins=[make_plugin_propagate()],
    )
    assert pred_tokens == ref_tokens


def test_tokensplit_split():
    ref_tokens = "A B CD E".split()
    user_tokens = "A B C D E".split()
    pred_tokens = decrypt_tokens(
        encrypt_tokens(ref_tokens),
        user_tokens,
        decryption_plugins=[make_plugin_split(8, 8)],
    )
    assert pred_tokens == ref_tokens


def test_tokenmerge_split():
    ref_tokens = "A B C D E".split()
    user_tokens = "A B CD E".split()
    pred_tokens = decrypt_tokens(
        encrypt_tokens(ref_tokens),
        user_tokens,
        decryption_plugins=[make_plugin_split(8, 8)],
    )
    assert pred_tokens == ref_tokens


def test_deletion():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C E E".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), user_tokens)
    assert pred_tokens == "A B C [UNK] E E".split()


def test_addition():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C X D E E".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), user_tokens)
    assert pred_tokens == ref_tokens


def block_input():
    ref_tokens = "A B C D E".split()
    pred_tokens = decrypt_tokens([ref_tokens, ref_tokens], [ref_tokens, ref_tokens])
    assert pred_tokens == ref_tokens


@given(st.lists(st.text()))
def test_encrypt_decrypt_recover_original_tokens(tokens: list[str]):
    assert decrypt_tokens(encrypt_tokens(tokens), tokens) == tokens


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_propagate_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    encrypted = encrypt_tokens(tokens, hash_len=hash_len)
    decrypted = decrypt_tokens(encrypted, error_tokens, hash_len=hash_len)
    decrypted_with_propagate = decrypt_tokens(
        encrypted,
        error_tokens,
        hash_len=hash_len,
        decryption_plugins=[make_plugin_propagate()],
    )
    assert errors_nb(tokens, decrypted_with_propagate) <= errors_nb(tokens, decrypted)


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_split_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    encrypted = encrypt_tokens(tokens, hash_len=hash_len)
    decrypted = decrypt_tokens(encrypted, error_tokens, hash_len=hash_len)
    decrypted_with_propagate = decrypt_tokens(
        encrypted,
        error_tokens,
        hash_len=hash_len,
        decryption_plugins=[make_plugin_split(max_token_len=24, max_splits_nb=4)],
    )
    assert errors_nb(tokens, decrypted_with_propagate) <= errors_nb(tokens, decrypted)


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_case_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    encrypted = encrypt_tokens(tokens, hash_len=hash_len)
    decrypted = decrypt_tokens(encrypted, error_tokens, hash_len=hash_len)
    decrypted_with_propagate = decrypt_tokens(
        encrypted,
        error_tokens,
        hash_len=hash_len,
        decryption_plugins=[make_plugin_case()],
    )
    assert errors_nb(tokens, decrypted_with_propagate) <= errors_nb(tokens, decrypted)
