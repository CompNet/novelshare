from typing import Optional, Generator, Any, Literal, Callable
import re
import pathlib as pl, itertools as it
from dataclasses import dataclass
from more_itertools import flatten
from datasets import load_dataset as hf_load_dataset, VerificationMode
from sacred.run import Run
from novelties_bookshare.conll import load_conll2002_bio
from novelties_bookshare.utils import CorefMention
from novelties_bookshare.experiments.metrics import (
    log_ner_task_metrics_,
    log_coref_task_metrics_,
)

EDITION_SETS = {
    "Frankenstein": {
        "F-1818": "./data/Frankenstein/F-1818",
        "F-1823": "./data/Frankenstein/F-1823",
        "F-1831": "./data/Frankenstein/F-1831",
    },
    "Moby_Dick": {
        "MD-1851-US": "./data/Moby_Dick/MD-1851-US",
        "MD-1851-UK": "./data/Moby_Dick/MD-1851-UK",
        "MD-1988": "./data/Moby_Dick/MD-1988",
    },
    "Pride_and_Prejudice": {
        "PP-1813": "./data/Pride_and_Prejudice/PP-1813",
        "PP-1817": "./data/Pride_and_Prejudice/PP-1817",
        "PP-1894": "./data/Pride_and_Prejudice/PP-1894",
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


AlignFn = Callable[
    # args:
    [
        # hashed_chapters
        list[list[str]],
        # user_chapters
        list[list[str]],
        # hash_len
        int | None,
    ],
    # returns: aligned tokens
    list[str],
]


@dataclass
class Strategy:
    name: str
    align_fn: AlignFn


@dataclass
class Document:
    name: str
    chapters: list[list[str]]
    annotations: list[list[Any]] | None = None

    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        pass


CorpusID = Literal["3novels", "conll2003", "wnut2017", "litbank"]


class Conll2003Document(Document):
    ID2LABEL = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        assert not self.annotations is None
        ref_tokens = list(flatten(self.chapters))
        ref_tags = [self.ID2LABEL[tag_id] for tag_id in flatten(self.annotations)]
        log_ner_task_metrics_(_run, setup_name, ref_tokens, ref_tags, aligned_tokens)


class WNUT2017Document(Document):
    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        assert not self.annotations is None
        ref_tokens = list(flatten(self.chapters))
        ref_tags = list(flatten(self.annotations))
        log_ner_task_metrics_(_run, setup_name, ref_tokens, ref_tags, aligned_tokens)


class CoreferenceDocument(Document):
    @staticmethod
    def from_coref_data(
        name: str, sentences: list[list[str]], coref_chains: list[list[list[int]]]
    ) -> "CoreferenceDocument":
        tokens = [token for sent in sentences for token in sent]

        # in the case of coreference resolution, an annotation consist
        # in a list of Mention the tokens is part of.
        annotations = [[] for _ in tokens]
        sent_start = list(
            it.accumulate([0] + [len(sent) for sent in sentences], lambda x, y: x + y)
        )
        for chain_id, chain in enumerate(coref_chains):
            for sent_i, start, end in chain:
                token_start = sent_start[sent_i] + start
                token_end = sent_start[sent_i] + end
                for token_i in range(token_start, token_end + 1):
                    annotations[token_i].append(
                        CorefMention(token_start, token_end, chain_id)
                    )

        return CoreferenceDocument(name, [tokens], [annotations])

    def log_alignment_task_metrics(
        self, _run: Run, setup_name: str, aligned_tokens: list[str]
    ):
        assert not self.annotations is None
        ref_tokens = list(flatten(self.chapters))
        log_coref_task_metrics_(
            _run, setup_name, ref_tokens, self.annotations, aligned_tokens
        )


def load_corpus(name: CorpusID, **kwargs) -> list[Document]:
    if name == "3novels":
        return [
            Document(
                "F-1818",
                list(iter_book_chapters("./data/Frankenstein/F-1818", **kwargs)),
            ),
            Document(
                "MD-1851-US",
                list(iter_book_chapters("./data/Moby_Dick/MD-1851-US", **kwargs)),
            ),
            Document(
                "PP-1813",
                list(
                    iter_book_chapters("./data/Pride_and_Prejudice/PP-1813", **kwargs)
                ),
            ),
        ]
    elif name == "conll2003":
        conll2003 = hf_load_dataset(
            "BramVanroy/conll2003",
            revision="4ffbd53d9e0b92b473b9b7dcff12f53e7c17ce0c",
            verification_mode=VerificationMode.ALL_CHECKS,
        )
        return [
            Conll2003Document(
                split,
                [row["tokens"] for row in conll2003[split]],
                annotations=[row["ner_tags"] for row in conll2003[split]],
            )
            for split in ["train", "validation", "test"]
        ]
    elif name == "wnut2017":
        wnut2017 = hf_load_dataset(
            "extraordinarylab/wnut2017",
            revision="a2495caff3e288bd7640cbdba313dc76a75c5c4a",
            verification_mode=VerificationMode.ALL_CHECKS,
        )
        return [
            WNUT2017Document(
                split,
                [row["tokens"] for row in wnut2017[split]],
                annotations=[row["ner_tags"] for row in wnut2017[split]],
            )
            for split in ["train", "validation", "test"]
        ]
    elif name == "litbank":
        litbank = hf_load_dataset(
            "coref-data/litbank_raw",
            "split_0",
            revision="14cac705d08a68f1df8eb197b57a9f98ae920e54",
            verification_mode=VerificationMode.ALL_CHECKS,
        )
        return [
            CoreferenceDocument.from_coref_data(
                row["doc_name"], row["sentences"], row["coref_chains"]
            )
            for split in ["train", "validation", "test"]
            for row in litbank[split]
        ]
    raise ValueError(name)
