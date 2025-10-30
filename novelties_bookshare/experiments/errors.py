import random, copy
import numpy as np
from collections import Counter
from scrambledtext import CorruptionEngine
from novelties_bookshare.experiments.ocr_utils import (
    DEFAULT_PROBABILITY_DISTRIBUTIONS,
    _load_ocr_probability_distributions_from_dict,
)

OCR_CORRUPTION_PROBS = _load_ocr_probability_distributions_from_dict(
    DEFAULT_PROBABILITY_DISTRIBUTIONS
)


def _token_dist(tokens: list[str]) -> dict[str, float]:
    assert len(tokens) > 0
    counter = Counter(tokens)
    return {token: count / len(tokens) for token, count in counter.items()}


def _sample(token_dist: dict[str, float]) -> str:
    assert len(token_dist) > 0
    vocab = list(token_dist.keys())
    weights = list(token_dist.values())
    return random.choices(vocab, weights=weights)[0]


def substitute(tokens: list[str], subst_nb: int) -> list[str]:
    token_dist = _token_dist(tokens)
    indices = np.random.choice(list(range(len(tokens))), subst_nb, replace=False)

    noisy_tokens = copy.deepcopy(tokens)
    for i in indices:
        noisy_tokens[i] = _sample(token_dist)

    return noisy_tokens


def delete(tokens: list[str], deletion_nb: int) -> list[str]:
    indices = set(
        np.random.choice(list(range(len(tokens))), deletion_nb, replace=False)
    )
    return [tok for i, tok in enumerate(tokens) if not i in indices]


def add(tokens: list[str], addition_nb: int) -> list[str]:
    token_dist = _token_dist(tokens)
    noisy_tokens = copy.deepcopy(tokens)
    for _ in range(addition_nb):
        noisy_tokens.insert(
            random.randint(0, len(noisy_tokens) - 1), _sample(token_dist)
        )
    return noisy_tokens


def ocr_scramble(tokens: list[str], wer: float, cer: float) -> list[str]:
    if cer == 0.0:
        return tokens

    engine = CorruptionEngine(
        OCR_CORRUPTION_PROBS.conditional,
        OCR_CORRUPTION_PROBS.substitutions,
        OCR_CORRUPTION_PROBS.insertions,
        target_wer=wer,
        target_cer=cer,
    )

    corrupted_text, _, _, _ = engine.corrupt_text(" ".join(tokens))
    corrupted_tokens = corrupted_text.split()
    return corrupted_tokens  # type: ignore


def token_split(tokens: list[str], split_nb: int) -> list[str]:
    split_indices = set(
        np.random.choice(list(range(len(tokens))), split_nb, replace=False)
    )

    noisy_tokens = []
    for i, token in enumerate(tokens):
        if i in split_indices:
            if len(token) >= 2:
                split_idx = random.randint(1, len(token))
                noisy_tokens.append(token[:split_idx])
                noisy_tokens.append(token[split_idx:])
        else:
            noisy_tokens.append(token)

    return noisy_tokens


def token_merge(tokens: list[str], merge_nb: int) -> list[str]:
    merge_indices = set(
        np.random.choice(list(range(len(tokens) - 1)), merge_nb, replace=False)
    )

    noisy_tokens = []
    for i, tok in enumerate(tokens):
        if i - 1 in merge_indices:
            noisy_tokens[-1] += tok
        else:
            noisy_tokens.append(tok)

    return noisy_tokens
