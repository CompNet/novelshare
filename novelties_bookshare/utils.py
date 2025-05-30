from typing import Generator, Iterable, TypeVar, List
from more_itertools import windowed

R = TypeVar("R")


def iterate_pattern(seq: Iterable[R], pattern: List[R]) -> Generator[int, None, None]:
    """Search a pattern in sequence

    :param seq: sequence in which to search
    :param pattern: searched pattern
    :return: a list of patterns start index
    """
    for subseq_i, subseq in enumerate(windowed(seq, len(pattern))):
        if list(subseq) == pattern:
            yield subseq_i


def strksplit(string: str, k: int, _i: int = 1) -> list[list[str]]:
    """Enumerate all possible ways of splitting a string into k
    substrings using backtracking.  Do not count empty strings as a
    valid substring.

    >>> strksplit("abc", 2)
    [['a', 'bc'], ['ab', 'c']]

    :param string: the string to split
    :param k: the number of substring to generate
    :param _i: private parameter, indicating the current split
        decision index
    :return: a list of all possible splits of STRNIG of size K
    """
    assert k >= 1
    splits = []

    if k == 1:
        return [[string]]

    if _i >= len(string):
        return []

    # 1. we choose to split here
    for split in strksplit(string[_i:], k - 1, 1):
        splits.append([string[:_i]] + split)

    # 2. we do not split here
    splits += strksplit(string, k, _i + 1)

    return splits
