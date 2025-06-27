import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio


def load_book(path: pl.Path) -> tuple[list[str], list[str]]:
    tokens = []
    tags = []
    for path in sorted(path.glob("*.conll")):
        chapter_tokens, chapter_tags = load_conll2002_bio(str(path))
        tokens += chapter_tokens
        tags += chapter_tags
    return tokens, tags
