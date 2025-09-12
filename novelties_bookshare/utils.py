from typing import Generator, Iterable, TypeVar, Optional
from dataclasses import dataclass
from more_itertools import windowed

R = TypeVar("R")


def iterate_pattern(seq: Iterable[R], pattern: list[R]) -> Generator[int, None, None]:
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


@dataclass
class NEREntity:
    tokens: list[str]
    start: int
    end: int
    tag: str


def ner_entities(
    tokens: list[str], bio_tags: list[str], resolve_inconsistencies: bool = True
) -> list[NEREntity]:
    """Extract NER entities from a list of BIO tags

    .. note::

        adapted from Renard

    :param tokens: a list of tokens
    :param bio_tags: a list of BIO tags.  In particular, BIO tags
        should be in the CoNLL-2002 form (such as 'B-PER I-PER')

    :return: A list of ner entities, in apparition order
    """
    assert len(tokens) == len(bio_tags)

    entities = []
    current_tag: Optional[str] = None
    current_tag_start_idx: Optional[int] = None

    for i, tag in enumerate(bio_tags):
        if not current_tag is None and not tag.startswith("I-"):
            assert not current_tag_start_idx is None
            entities.append(
                NEREntity(
                    tokens[current_tag_start_idx:i],
                    current_tag_start_idx,
                    i,
                    current_tag,
                )
            )
            current_tag = None
            current_tag_start_idx = None

        if tag.startswith("B-"):
            current_tag = tag[2:]
            current_tag_start_idx = i

        elif tag.startswith("I-"):
            if current_tag is None and resolve_inconsistencies:
                current_tag = tag[2:]
                current_tag_start_idx = i
                continue

    if not current_tag is None:
        assert not current_tag_start_idx is None
        entities.append(
            NEREntity(
                tokens[current_tag_start_idx : len(tokens)],
                current_tag_start_idx,
                len(bio_tags),
                current_tag,
            )
        )

    return entities
