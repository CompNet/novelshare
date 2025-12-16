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

    X = list(range(1, 65))
    Y = []
    for hash_len in tqdm(X, ascii=True):
        hash2tokens = defaultdict(set)
        encrypted = encrypt_tokens(tokens, hash_len=hash_len)
        for e, token in zip(encrypted, tokens):
            hash2tokens[e].add(token)
        Y.append(mean(len(v) - 1 for v in hash2tokens.values()))

    print(Y)
    assert all(Y[i] >= Y[i + 1] for i in range(len(Y) - 1))
    Y = [value for value in Y if value > 0.01]

    plt.style.use("science")
    plt.rcParams.update({"font.size": 10})
    X = [i + 1 for i in range(len(Y))]
    plt.plot(X, Y, linewidth=1, marker="*", markersize=8)
    ax = plt.gca()
    for x, y in zip(X, Y):
        ax.annotate(
            f"{y:.2f}",
            (x, y + 24),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid()
    ax.set_ylim((0, max(Y) + 256))
    plt.ylabel("Mean collisions per token", fontsize=10)
    plt.xlabel("Hash length")
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches(4, 2)
    if not args.output_file is None:
        plt.savefig(args.output_file)
    else:
        plt.show()
