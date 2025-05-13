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
