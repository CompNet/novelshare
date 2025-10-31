from typing import Optional, Generator
import re
import pathlib as pl
from novelties_bookshare.conll import load_conll2002_bio

EDITION_SETS = {
    "Brave_New_World": {
        "HC98": "./data/Brave_New_World/HC98",
        "HC06": "./data/Brave_New_World/HC06",
        "HC04": "./data/Brave_New_World/HC04",
        "RB06": "./data/Brave_New_World/RB06",
    },
    "Frankenstein": {
        "PG84": "./data/Frankenstein/PG84",
        # we do not use PG41445 as it does not have the same number of
        # chapters as the other two.
        # "PG41445": "./data/Frankenstein/PG41445",
        "PG42324": "./data/Frankenstein/PG42324",
    },
    "Moby_Dick": {
        "PG15": "./data/Moby_Dick/PG15",
        "PG2489": "./data/Moby_Dick/PG2489",
        "PG2701": "./data/Moby_Dick/PG2701",
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
