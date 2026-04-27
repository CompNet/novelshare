from hypothesis import given, strategies as st
from novelshare.hash import hash_tokens
from novelshare.align import (
    align_tokens,
    make_plugin_case,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_retokenize,
)
from novelshare.experiments.metrics import errors_nb
from tests.strategies import error_seq_pairs


def test_substitution():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    pred_tokens = align_tokens(hash_tokens(ref_tokens), user_tokens)
    assert pred_tokens == "A B C D E [UNK]".split()


def test_substitution_propagate():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D E X".split()
    pred_tokens = align_tokens(
        hash_tokens(ref_tokens),
        user_tokens,
        alignment_plugins=[make_plugin_propagate()],
    )
    assert pred_tokens == ref_tokens


def test_tokensplit_retokenize():
    ref_tokens = "A B CD E".split()
    user_tokens = "A B C D E".split()
    pred_tokens = align_tokens(
        hash_tokens(ref_tokens),
        user_tokens,
        alignment_plugins=[make_plugin_retokenize(8, 8)],
    )
    assert pred_tokens == ref_tokens


def test_tokenmerge_retokenize():
    ref_tokens = "A B C D E".split()
    user_tokens = "A B CD E".split()
    pred_tokens = align_tokens(
        hash_tokens(ref_tokens),
        user_tokens,
        alignment_plugins=[make_plugin_retokenize(8, 8)],
    )
    assert pred_tokens == ref_tokens


def test_deletion():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C E E".split()
    pred_tokens = align_tokens(hash_tokens(ref_tokens), user_tokens)
    assert pred_tokens == "A B C [UNK] E E".split()


def test_addition():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C X D E E".split()
    pred_tokens = align_tokens(hash_tokens(ref_tokens), user_tokens)
    assert pred_tokens == ref_tokens


def test_block_input():
    ref_tokens = "A B C D E".split()
    hashed_tokens = hash_tokens(ref_tokens)
    pred_tokens = align_tokens([hashed_tokens, hashed_tokens], [ref_tokens, ref_tokens])
    assert pred_tokens == ref_tokens + ref_tokens


@given(st.lists(st.text()))
def test_hash_align_recover_original_tokens(tokens: list[str]):
    assert align_tokens(hash_tokens(tokens), tokens) == tokens


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_propagate_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    hashed = hash_tokens(tokens, hash_len=hash_len)
    aligned = align_tokens(hashed, error_tokens, hash_len=hash_len)
    aligned_with_propagate = align_tokens(
        hashed,
        error_tokens,
        hash_len=hash_len,
        alignment_plugins=[make_plugin_propagate()],
    )
    assert errors_nb(tokens, aligned_with_propagate) <= errors_nb(tokens, aligned)


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_retokenize_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    hashed = hash_tokens(tokens, hash_len=hash_len)
    aligned = align_tokens(hashed, error_tokens, hash_len=hash_len)
    aligned_with_propagate = align_tokens(
        hashed,
        error_tokens,
        hash_len=hash_len,
        alignment_plugins=[make_plugin_retokenize(max_token_len=24, max_splits_nb=4)],
    )
    assert errors_nb(tokens, aligned_with_propagate) <= errors_nb(tokens, aligned)


@given(error_seq_pairs(), st.integers(min_value=1, max_value=64))
def test_case_cant_degrade(error_pair: tuple[list[str], list[str]], hash_len):
    tokens, error_tokens = error_pair
    hashed = hash_tokens(tokens, hash_len=hash_len)
    aligned = align_tokens(hashed, error_tokens, hash_len=hash_len)
    aligned_with_propagate = align_tokens(
        hashed,
        error_tokens,
        hash_len=hash_len,
        alignment_plugins=[make_plugin_case()],
    )
    assert errors_nb(tokens, aligned_with_propagate) <= errors_nb(tokens, aligned)
