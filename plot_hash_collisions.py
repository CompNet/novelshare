from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt
from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.experiments.data import load_book

if __name__ == "__main__":
    tokens, _ = load_book("./data/editions_diff/Moby_Dick/Novelties")

    x = list(range(1, 65))
    y = []
    for hash_len in x:
        hash2tokens = defaultdict(set)
        encrypted = encrypt_tokens(tokens, hash_len=hash_len)
        for e, token in zip(encrypted, tokens):
            hash2tokens[e].add(token)
        y.append(mean(len(v) - 1 for v in hash2tokens.values()))

    print(y)
    plt.plot(x, y)
    for xi, yi in zip(x, y):
        if yi <= 0.01:
            continue
        plt.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color="red")
    plt.ylabel("Mean number of hash collisions")
    plt.xlabel("Hash length")
    plt.grid()
    plt.show()
