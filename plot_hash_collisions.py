import argparse
import pathlib as pl
from collections import defaultdict
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.experiments.data import load_book, EDITION_SETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-file", type=pl.Path, default=None)
    args = parser.parse_args()

    tokens = []
    for novel, edition_sets in EDITION_SETS.items():
        for path in edition_sets.values():
            tokens += load_book(path)

    x = list(range(1, 65))
    y = []
    for hash_len in tqdm(x, ascii=True):
        hash2tokens = defaultdict(set)
        encrypted = encrypt_tokens(tokens, hash_len=hash_len)
        for e, token in zip(encrypted, tokens):
            hash2tokens[e].add(token)
        y.append(mean(len(v) - 1 for v in hash2tokens.values()))

    print(y)
    assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))
    y = [value for value in y if value > 0.01]

    plt.style.use("science")
    plt.rcParams.update({"font.size": 46})
    plt.bar([str(i + 1) for i in range(len(y))], y)
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=42,
        )
    ax.set_ylim((0, max(y) + 256))
    plt.ylabel("Mean collisions per token", fontsize=36)
    plt.xlabel("Hash length")
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    if not args.output_file is None:
        plt.savefig(args.output_file)
    else:
        plt.show()
