from typing import Union
import re, argparse, json
from collections import defaultdict
import pathlib as pl
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd


def get_params(metric_key: str) -> tuple[str, dict[str, str]]:
    # form of each metric
    # s=strat.e=edition.error_nb
    m = re.match(r"s=([^\.]+)\.e=([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    strat, edition, metric_name = m.groups()
    return metric_name, {"strategy": strat, "edition": edition}


def format_bar_height(bar_value: Union[int, float]) -> str:
    if isinstance(bar_value, float):
        return f"{bar_value:.2f}"
    return str(bar_value)


METRIC_TO_YLABEL = {
    "errors_nb": "Number of errors",
    "duration_s": "Duration in seconds",
    "errors_percent": "Percentage of errors",
    "entity_errors_nb": "Number of entity errors",
    "entity_errors_percent": "Percentage of entity errors",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=pl.Path)
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        help="one of: 'errors_nb', 'duration_s', 'errors_percent', 'entity_errors_nb', 'entity_errors_percent'",
    )
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    args = parser.parse_args()

    df_dict = defaultdict(list)
    with open(args.run / "metrics.json") as f:
        data = json.load(f)

        lines = defaultdict(dict)
        for key, metric_dict in data.items():
            metric_name, params = get_params(key)
            params_key = (
                params["strategy"],
                params["edition"],
            )
            lines[params_key][metric_name] = metric_dict["values"][0]

        for (strat, edition), metric_dict in lines.items():
            df_dict["strategy"].append(strat)
            df_dict["edition"].append(edition)
            for key, value in metric_dict.items():
                df_dict[key].append(value)

    df = pd.DataFrame(df_dict)
    print(df)

    df = df.pivot(index="edition", columns="strategy", values=args.metric)
    df = df.reset_index().set_index("edition")
    df = df[df.mean().sort_values(ascending=False).index]

    plt.style.use("science")
    plt.rcParams.update({"font.size": 16})
    ax = df.plot.bar()
    for p in ax.patches:
        ax.annotate(
            format_bar_height(p.get_height()),
            (p.get_x() * 1.005, p.get_height() * 1.005),
            fontsize=12,
        )
    ax.set_ylabel(METRIC_TO_YLABEL[args.metric])
    plt.grid()
    plt.show()
