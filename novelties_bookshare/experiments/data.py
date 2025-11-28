from typing import Optional, Generator
import re
import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio

EDITION_SETS = {
    "Frankenstein": {
        "F-1818": "./data/Frankenstein/F-1818",
        "F-1823": "./data/Frankenstein/F-1823",
        "F-1831": "./data/Frankenstein/F-1831",
    },
    "Moby_Dick": {
        "MB-1851-US": "./data/Moby_Dick/MB-1851-US",
        "MB-1851-UK": "./data/Moby_Dick/MB-1851-UK",
        "MB-1988": "./data/Moby_Dick/MB-1988",
    },
    "Pride_and_Prejudice": {
        "PG1342": "./data/PrideAndPrejudice/PG1342",
        "PG42671": "./data/PrideAndPrejudice/PG42671",
    },
}


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


def replace_(chapters: list[list[str]], replacements: list[tuple[list[str], str]]):
    for chapter in chapters:
        for i, token in enumerate(chapter):
            for repl_source, repl_target in replacements:
                if token in repl_source:
                    chapter[i] = repl_target


def normalize_(chapters: list[list[str]]):
    replace_(chapters, [(["``", "''", "“", "”"], '"')])
    replace_(chapters, [(["‘", "’"], "'")])
    replace_(chapters, [(["…"], "...")])
    replace_(chapters, [(["—"], "-")])
