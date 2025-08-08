import re
import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio


def load_book(path: pl.Path | str) -> tuple[list[str], list[str]]:
    if isinstance(path, str):
        path = pl.Path(path)

    tokens = []
    tags = []
    chapter_paths = path.glob("chapter_*.conll")
    chapter_paths = sorted(
        chapter_paths,
        key=lambda p: int(re.match(r"chapter_([0-9]+)\.conll", str(p.name)).group(1)),
    )

    for path in chapter_paths:
        chapter_tokens, chapter_tags = load_conll2002_bio(str(path))
        tokens += chapter_tokens
        tags += chapter_tags

    return tokens, tags
