from novelties_bookshare.experiments.errors import (
    substitute,
    delete,
    add,
    _sample,
    token_merge,
    token_split,
)
from hypothesis import assume, given, strategies as st


def test_sample_one_prob():
    assert _sample({"A": 0.0, "B": 1.0}) == "B"


@given(st.lists(st.text(), min_size=1), st.integers(min_value=1, max_value=1024))
def test_delete_len(tokens: list[str], del_nb: int):
    assume(del_nb <= len(tokens))
    deleted = delete(tokens, 1)
    assert len(deleted) == len(tokens) - 1


@given(st.lists(st.text(), min_size=1), st.integers(min_value=1, max_value=1024))
def test_add_len(tokens: list[str], add_nb: int):
    added = add(tokens, add_nb)
    assert len(added) == len(tokens) + add_nb


@given(st.lists(st.text(), min_size=0), st.integers(min_value=1, max_value=1024))
def test_substitute_len(tokens: list[str], subst_nb: int):
    assume(subst_nb <= len(tokens))
    substituted = substitute(tokens, subst_nb)
    assert len(substituted) == len(tokens)


@given(
    st.lists(st.text(min_size=2), min_size=1), st.integers(min_value=1, max_value=1024)
)
def test_token_split_len(tokens: list[str], split_nb: int):
    assume(split_nb <= len(tokens))
    split = token_split(tokens, split_nb)
    assert len(split) == len(tokens) + split_nb


@given(st.lists(st.text(), min_size=2), st.integers(min_value=1, max_value=1024))
def test_token_merge_len(tokens: list[str], merge_nb: int):
    assume(merge_nb < len(tokens))
    merged = token_merge(tokens, merge_nb)
    assert len(merged) == len(tokens) - merge_nb
