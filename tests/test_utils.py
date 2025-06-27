from hypothesis import given, strategies as st
import math
from novelties_bookshare.utils import strksplit


@given(st.text(max_size=16), st.integers(min_value=1, max_value=8))
def test_strksplit_has_correct_splits(string: str, k: int):
    assert all([len(split) == k for split in strksplit(string, k)])


@given(st.text(min_size=1, max_size=16), st.integers(min_value=1, max_value=8))
def test_strksplit_has_correct_number_of_splits(string: str, k: int):
    assert len(strksplit(string, k)) == math.comb(len(string) - 1, k - 1)


@given(st.text(max_size=16), st.integers(min_value=1, max_value=8))
def test_strksplit_has_valid_substring(string: str, k: int):
    assert all(
        [substring in string for split in strksplit(string, k) for substring in split]
    )
