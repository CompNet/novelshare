from typing import Union
import re, argparse, json
from collections import defaultdict
import pathlib as pl
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from novelties_bookshare.experiments.plot_utils import STRAT_COLOR_HINTS


def get_params(metric_key: str) -> tuple[str, dict[str, str]]:
    # s=strat.e=novel,edition.error_nb
    m = re.match(r"s=([^\.]+)\.e=([^\,]+),([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    strat, novel, edition, metric_name = m.groups()
    return metric_name, {
        "strategy": strat,
        "novel": novel,
        "edition": edition,
    }


def get_params_mlm(metric_key: str) -> tuple[str, dict[str, str]]:
    # w=window.e=novel,edition.metric_name
    m = re.match(r"w=([^\.]+)\.e=([^\,]+),([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    window, novel, edition, metric_name = m.groups()
    return metric_name, {"window": window, "novel": novel, "edition": edition}


def get_params_split(metric_key: str) -> tuple[str, dict[str, str]]:
    # t=max_token_len.s=max_split_nb.e=edition.metric_name
    m = re.match(r"t=([^\.]+)\.s=([^\.]+)\.e=([^\,]+),([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    max_token_len, max_split_nb, novel, edition, metric_name = m.groups()
    return metric_name, {
        "max_token_len": max_token_len,
        "max_split_nb": max_split_nb,
        "novel": novel,
        "edition": edition,
    }


def get_params_propagate(metric_key: str) -> tuple[str, dict[str, str]]:
    # p=pipeline.e=edition
    m = re.match(r"p=([^\.]+)\.e=([^\,]+),([^\.]+)\.(.*)", metric_key)
    if m is None:
        return "", {}
    pipeline, novel, edition, metric_name = m.groups()
    return metric_name, {"pipeline": pipeline, "edition": edition, "novel": novel}


def format_bar_height(bar_value: Union[int, float]) -> str:
    if isinstance(bar_value, float):
        return f"{bar_value:.2f}"
    return str(bar_value)


METRIC_TO_YLABEL = {
    "errors_nb": "Number of errors",
    "precision_errors_nb": "Number of precision errors",
    "duration_s": "Duration in seconds",
    "errors_percent": "Percentage of errors",
    "entity_errors_nb": "Number of entity errors",
    "entity_errors_percent": "Percentage of entity errors",
}

XP_PARAMS_KEY = {
    "xp_edition": ["strategy", "edition"],
    "xp_edition_mlm_params": ["window", "edition"],
    "xp_edition_split_params": ["max_token_len", "max_split_nb", "edition"],
    "xp_edition_propagate_order": ["pipeline", "edition"],
}

XP_GET_PARAMS_FN = {
    "xp_edition": get_params,
    "xp_edition_mlm_params": get_params_mlm,
    "xp_edition_split_params": get_params_split,
    "xp_edition_propagate_order": get_params_propagate,
}


def load_xp(path: pl.Path) -> tuple[str, pd.DataFrame]:
    with open(path / "run.json") as f:
        run_data = json.load(f)
    xp_name = run_data["experiment"]["name"]

    df_dict = defaultdict(list)
    with open(path / "metrics.json") as f:
        data = json.load(f)

        lines = defaultdict(dict)
        for key, metric_dict in data.items():
            metric_name, params = XP_GET_PARAMS_FN[xp_name](key)
            params_key = tuple(params[k] for k in XP_PARAMS_KEY[xp_name])
            lines[params_key][metric_name] = metric_dict["values"][0]

        for params, metric_dict in lines.items():
            for param_name, param_value in zip(XP_PARAMS_KEY[xp_name], params):
                df_dict[param_name].append(param_value)
            for key, value in metric_dict.items():
                df_dict[key].append(value)

    df = pd.DataFrame(df_dict)
    return xp_name, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runs",
        nargs="*",
        type=pl.Path,
        help="A list of runs to plot. They must be of same nature (i.e. obtained with the same experiment script).",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        help="one of: 'errors_nb', 'duration_s', 'errors_percent', 'entity_errors_nb', 'entity_errors_percent'",
    )
    parser.add_argument("-l", "--log-scale", action="store_true")
    parser.add_argument("-a", "--annotate-values", action="store_true")
    parser.add_argument("-o", "--output-file", type=pl.Path, default=None)
    args = parser.parse_args()

    assert len(args.runs) > 0
    xp_name, df = load_xp(args.runs[0])
    for run in args.runs[1:]:
        run_xp_name, run_df = load_xp(run)
        df = pd.concat([df, run_df])
    print(f"{xp_name=}")
    print(df)

    df = df.pivot(
        index="edition",
        columns=[k for k in XP_PARAMS_KEY[xp_name] if k != "edition"],
        values=args.metric,
    )
    df = df.reset_index().set_index("edition")
    df = df[df.mean().sort_values(ascending=False).index]
    try:  # if possible, sort columns in ascending order
        df = df.sort_index(axis=1, key=lambda x: x.astype(int))
    except ValueError:
        pass

    plt.style.use("science")
    plt.rcParams.update({"font.size": 10})
    ax = df.plot.bar(color=[STRAT_COLOR_HINTS[strat] for strat in df.columns])
    ax.legend(ncols=3)
    if args.annotate_values:
        for p in ax.patches:
            ax.annotate(
                format_bar_height(p.get_height()),
                (p.get_x() * 1.005, p.get_height() * 1.005),
                fontsize=8,
            )
    ax.set_xlabel("Edition")
    ax.set_ylabel(METRIC_TO_YLABEL[args.metric])
    ax.grid()
    if args.log_scale:
        ax.set_yscale("log")

    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    if not args.output_file is None:
        plt.savefig(args.output_file)
    else:
        plt.show()
