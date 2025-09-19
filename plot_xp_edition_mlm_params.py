import re, argparse, json
from collections import defaultdict
import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_params(metric_key: str) -> tuple[str, dict[str, str]]:
    # form of each metric
    # w=window.e=edition.metric_name
    m = re.match(r"w=([^\.]+)\.e=([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    window, edition, metric_name = m.groups()
    return metric_name, {"window": window, "edition": edition}


def pformat_number(number) -> str:
    if isinstance(number, float):
        return f"{number:.2f}"
    return str(number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=pl.Path)
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="errors_nb",
        help="one of 'errors_nb', 'duration_s'",
    )
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    args = parser.parse_args()

    df_dict = defaultdict(list)
    with open(args.run / "metrics.json") as f:
        data = json.load(f)

        lines = defaultdict(dict)
        for key, metric_dict in data.items():
            metric_name, params = get_params(key)
            lines[(params["window"], params["edition"])][metric_name] = metric_dict[
                "values"
            ][0]

        for (window, edition), metric_dict in lines.items():
            df_dict["window"].append(window)
            df_dict["edition"].append(edition)
            for key, value in metric_dict.items():
                df_dict[key].append(value)

    df = pd.DataFrame(df_dict)
    print(df)

    df = df.pivot(index="edition", columns="window", values=args.metric)
    df = df.reset_index().set_index("edition")
    df = df[df.mean().sort_values(ascending=False).index]
    ax = df.plot.bar()
    for p in ax.patches:
        ax.annotate(
            pformat_number(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
        )
    plt.show()
