from typing import Optional, Dict, List, Tuple
import os, sys


def load_conll2002_bio(
    path: Optional[str] = None, separator: str = " ", **kwargs
) -> Tuple[List[str], List[str]]:
    """Load a file under CoNLL2022 BIO format.

    :param path: path to the CoNLL-2002 formatted file.  If ``None``,
        read from stdin.
    :param separator: separator between token and BIO tags
    :param kwargs: additional kwargs for :func:`open` (such as
        ``encoding`` or ``newline``).

    :return: ``(tokens, tags)``
    """
    if not path is None:
        with open(os.path.expanduser(path), **kwargs) as f:
            raw_data = f.read()
    else:
        raw_data = sys.stdin.read()

    tokens = []
    tags = []
    for line in raw_data.split("\n"):
        line = line.strip("\n")
        if len(line) == 0:
            continue
        try:
            token, tag = line.split(separator)
        except ValueError:
            continue
        tokens.append(token)
        tags.append(tag)

    return tokens, tags


def dump_conll2002_bio(
    tokens: List[str],
    tags: List[str],
    path: Optional[str] = None,
    separator: str = " ",
    **kwargs,
):
    """Dump a list of tokens/tags in the CoNLL-2002 format.

    :param path: path to the output file.  If ``None``, print to stdout.
    :param separator: separator between token and BIO tags
    :param kwargs: additional kwargs for :func:`open` (such as
        ``encoding`` or ``newline``).
    """
    assert len(tokens) == len(tags)

    if not path is None:
        with open(path, "w", **kwargs) as f:
            for token, tag in zip(tokens, tags):
                f.write(f"{token}{separator}{tag}\n")
    else:
        for token, tag in zip(tokens, tags):
            print(f"{token}{separator}{tag}")
