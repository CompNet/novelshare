import copy
from typing import Optional
from hypothesis import strategies as st


@st.composite
def add_seq_pairs(
    draw, lists_kwargs: Optional[dict] = None, text_kwargs: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    lists_kwargs = lists_kwargs or {}
    text_kwargs = text_kwargs or {}
    tokens = draw(st.lists(st.text(**text_kwargs), min_size=1, **lists_kwargs))
    add_tokens = copy.deepcopy(tokens)
    # NOTE: the max value is not mandatory, but more having more
    # additions than tokens form the original sequence is probably not
    # realistic
    addition_nb = draw(st.integers(min_value=1, max_value=len(tokens)))
    for _ in range(addition_nb):
        add_index = draw(st.integers(min_value=0, max_value=len(add_tokens) - 1))
        add_tokens.insert(add_index, "[ADD]")
    return tokens, add_tokens


@st.composite
def del_seq_pairs(
    draw, lists_kwargs: Optional[dict] = None, text_kwargs: Optional[dict] = None
):
    lists_kwargs = lists_kwargs or {}
    text_kwargs = text_kwargs or {}
    tokens = draw(st.lists(st.text(**text_kwargs), min_size=1, **lists_kwargs))
    deletion_nb = draw(st.integers(min_value=1, max_value=len(tokens)))
    del_indices = draw(
        st.sets(
            st.integers(min_value=0, max_value=len(tokens) - 1),
            min_size=1,
            max_size=deletion_nb,
        )
    )
    del_tokens = [t for t in tokens if not t in del_indices]
    return tokens, del_tokens


@st.composite
def sub_seq_pairs(
    draw, lists_kwargs: Optional[dict] = None, text_kwargs: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    lists_kwargs = lists_kwargs or {}
    text_kwargs = text_kwargs or {}
    tokens = draw(st.lists(st.text(**text_kwargs), min_size=1, **lists_kwargs))
    sub_tokens = copy.deepcopy(tokens)
    sub_nb = draw(st.integers(min_value=1, max_value=len(tokens)))
    sub_set = draw(
        st.sets(
            st.integers(min_value=0, max_value=len(sub_tokens) - 1),
            min_size=1,
            max_size=sub_nb,
        ),
    )
    for i in sub_set:
        sub_tokens[i] = "[SUB]"
    return tokens, sub_tokens


@st.composite
def error_seq_pairs(draw):
    return draw(st.one_of(add_seq_pairs(), del_seq_pairs(), sub_seq_pairs()))
