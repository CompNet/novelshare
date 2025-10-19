from collections import defaultdict
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.experiments.data import load_book

if __name__ == "__main__":
    tokens = (
        load_book("./data/editions_diff/Moby_Dick/PG15/")
        + load_book("./data/editions_diff/Moby_Dick/PG2489/")
        + load_book("./data/editions_diff/Moby_Dick/PG2701/")
    )

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
    plt.rcParams.update({"font.size": 42})
    plt.bar([str(i + 1) for i in range(len(y))], y)
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=40,
        )
    plt.ylabel("Mean number of hash collisions")
    plt.xlabel("Hash length")
    plt.show()
