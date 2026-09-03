from datasets import Dataset
from novelshare.align import align_tokens, AlignmentPlugin
from novelshare.hash import hash_tokens


def align_hf_dataset_tokens(
    dataset: Dataset,
    user_tokens: list[list[str]],
    hash_len: int | None = None,
    alignment_plugins: list[AlignmentPlugin] | None = None,
    tokens_col: str = "tokens",
) -> Dataset:
    """Attempt to align tokens with annotations using the provided
    user tokens.

    :param dataset: dataset for which to ailgn tokens.
    :param user_tokens: user tokens.
    :param hash_len: passed to :func:`.align_tokens`.
    :param alignment_plugins: passed to :func:`.align_tokens`.
    :param tokens_col: the column containing the tokens to align.

    :return: a new dataset, where the tokens column is replaced by the
             aligned tokens.
    """
    assert len(dataset) == len(user_tokens)

    def align_row_tokens(row: dict, i: int) -> dict:
        row[tokens_col] = align_tokens(
            row[tokens_col],
            user_tokens[i],
            hash_len=hash_len,
            alignment_plugins=alignment_plugins,
        )
        return row

    return dataset.map(align_row_tokens, with_indices=True)


def hash_hf_dataset_tokens(
    dataset: Dataset, hash_len: int | None = None, tokens_col: str = "tokens"
) -> Dataset:
    """Hash the tokens of a huggingface dataset.

    :param dataset: dataset for which to hash tokens.
    :param hash_len: passed to :func:`.hash_tokens`.
    :param tokens_col: the column containing the tokens to hash.
    """

    def hash_row_tokens(row: dict) -> dict:
        row[tokens_col] = hash_tokens(row[tokens_col], hash_len=hash_len)
        return row

    return dataset.map(hash_row_tokens)
