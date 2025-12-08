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
    ref_tokens: list[str], pred_tokens: list[str], ref_tags: list[str]
) -> int:
    entities = ner_entities(ref_tokens, ref_tags, resolve_inconsistencies=True)
    errors_nb = 0
    for entity in entities:
        if (
            ref_tokens[entity.start : entity.end]
            != pred_tokens[entity.start : entity.end]
        ):
            errors_nb += 1
    return errors_nb


def entity_errors_percent(
    ref_tokens: list[str], pred_tokens: list[str], ref_tags: list[str]
) -> float:
    entities = ner_entities(ref_tokens, ref_tags, resolve_inconsistencies=True)
    if len(entities) == 0:
        return 0
    errors_nb = entity_errors_nb(ref_tokens, pred_tokens, ref_tags)
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

    # TODO: these metrics should be added back when implementing
    # general annotations
    # _run.log_scalar(
    #     f"{setup_name}.entity_errors_nb",
    #     entity_errors_nb(ref_tokens, pred_tokens, ref_tags),
    # )
    # _run.log_scalar(
    #     f"{setup_name}.entity_errors_percent",
    #     entity_errors_percent(ref_tokens, pred_tokens, ref_tags),
    # )

    _run.log_scalar(f"{setup_name}.duration_s", duration_s)
