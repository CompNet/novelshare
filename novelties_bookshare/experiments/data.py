from typing import Optional, Generator
import re
import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio


def iter_book_chapters(
    path: pl.Path | str, chapter_limit: Optional[int] = None
) -> Generator[list[str], None, None]:
    if isinstance(path, str):
        path = pl.Path(path)
    path = path.expanduser()

    chapter_paths = path.glob("chapter_*.conll")
    chapter_paths = sorted(
        chapter_paths,
        key=lambda p: int(re.match(r"chapter_([0-9]+)\.conll", str(p.name)).group(1)),
    )
    if chapter_paths is not None:
        chapter_paths = chapter_paths[:chapter_limit]

    for path in chapter_paths:
        chapter_tokens, _ = load_conll2002_bio(str(path))
        yield chapter_tokens


def load_book(path: pl.Path | str, chapter_limit: Optional[int] = None) -> list[str]:
    tokens = []
    for chapter_tokens in iter_book_chapters(path, chapter_limit):
        tokens += chapter_tokens
    return tokens
