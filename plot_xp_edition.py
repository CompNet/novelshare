import re, argparse, json
from collections import defaultdict
import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_params(metric_key: str) -> dict[str, str]:
    # form of each metric
    # s=strat.e=edition.error_nb
    m = re.match(r"s=([^\.]+)\.e=([^\.]+)\..*", metric_key)
    if m is None:
        return {}
    strat, edition = m.groups()
    return {"strat": strat, "edition": edition}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=pl.Path)
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    args = parser.parse_args()

    df_dict = defaultdict(list)
    with open(args.run / "metrics.json") as f:
        data = json.load(f)
        for key, metric_dict in data.items():
            params = get_params(key)
            df_dict["strategy"].append(params["strat"])
            df_dict["edition"].append(params["edition"])
            df_dict["error_nb"].append(metric_dict["values"][0])
    df = pd.DataFrame(df_dict)

    df = df.pivot(index="edition", columns="strategy", values="error_nb")
    df = df.reset_index().set_index("edition")
    df = df[df.mean().sort_values(ascending=False).index]
    df.plot.bar()
    plt.show()
