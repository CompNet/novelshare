from typing import Literal
from collections import defaultdict
from sacred.run import Run
from novelties_bookshare.utils import ner_entities


def errors_nb(ref_tokens: list[str], pred_tokens: list[str]) -> int:
    return sum(1 if ref != pred else 0 for ref, pred in zip(ref_tokens, pred_tokens))


def errors_percent(ref_tokens: list[str], pred_tokens: list[str]) -> float:
    if len(ref_tokens) == 0:
        return 0
    return errors_nb(ref_tokens, pred_tokens) / len(ref_tokens)


def entity_errors_nb(
    ref_tokens: list[str],
    pred_tokens: list[str],
    ref_tags: list[str],
    mode: Literal["lenient", "strict"],
) -> int:
    entities = ner_entities(ref_tokens, ref_tags, resolve_inconsistencies=True)
    errors_nb = 0
    for entity in entities:
        if mode == "strict":
            if (
                ref_tokens[entity.start : entity.end]
                != pred_tokens[entity.start : entity.end]
            ):
                errors_nb += 1
        elif mode == "lenient":
            if all(
                r != p
                for r, p in zip(
                    ref_tokens[entity.start : entity.end],
                    pred_tokens[entity.start : entity.end],
                )
            ):
                errors_nb += 1
        else:
            raise ValueError(mode)
    return errors_nb


def entity_errors_percent(
    ref_tokens: list[str],
    pred_tokens: list[str],
    ref_tags: list[str],
    mode: Literal["lenient", "strict"],
) -> float:
    entities = ner_entities(ref_tokens, ref_tags, resolve_inconsistencies=True)
    if len(entities) == 0:
        return 0
    errors_nb = entity_errors_nb(ref_tokens, pred_tokens, ref_tags, mode)
    return errors_nb / len(entities)


def precision_errors_nb(ref_tokens: list[str], pred_tokens: list[str]) -> float:
    return sum(
        1 if ref != pred and pred != "[UNK]" else 0
        for ref, pred in zip(ref_tokens, pred_tokens)
    )


def errors(ref_tokens: list[str], pred_tokens: list[str]) -> dict[str, list[str]]:
    """Record errors for each reference token

    :return: a dict of the form { ref_token : [ incorrect_pred_token,... ] }
    """
    error_dict = defaultdict(list)
    for ref, pred in zip(ref_tokens, pred_tokens):
        if ref != pred:
            error_dict[ref].append(pred)
    return error_dict


def record_decryption_metrics_(
    _run: Run,
    setup_name: str,
    ref_tokens: list[str],
    pred_tokens: list[str],
    duration_s: float,
    ref_tags: list[str] | None = None,
):
    _run.log_scalar(
        f"{setup_name}.errors_nb",
        errors_nb(ref_tokens, pred_tokens),
    )
    _run.log_scalar(
        f"{setup_name}.precision_errors_nb",
        precision_errors_nb(ref_tokens, pred_tokens),
    )
    _run.log_scalar(
        f"{setup_name}.errors_percent",
        errors_percent(ref_tokens, pred_tokens),
    )

    if not ref_tags is None:
        _run.log_scalar(
            f"{setup_name}.entity_errors_nb_lenient",
            entity_errors_nb(ref_tokens, pred_tokens, ref_tags, "lenient"),
        )
        _run.log_scalar(
            f"{setup_name}.entity_errors_percent_lenient",
            entity_errors_percent(ref_tokens, pred_tokens, ref_tags, "lenient"),
        )
        _run.log_scalar(
            f"{setup_name}.entity_errors_nb_strict",
            entity_errors_nb(ref_tokens, pred_tokens, ref_tags, "strict"),
        )
        _run.log_scalar(
            f"{setup_name}.entity_errors_percent_strict",
            entity_errors_percent(ref_tokens, pred_tokens, ref_tags, "strict"),
        )

    _run.log_scalar(f"{setup_name}.duration_s", duration_s)
