import argparse, json
from collections import defaultdict
import pathlib as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scienceplots
import pandas as pd
from more_itertools import windowed
from plot_xp_edition import get_params, METRIC_TO_YLABEL, XP_PARAMS_KEY
from novelties_bookshare.experiments.plot_utils import MARKERS

METRIC_TO_YFORMATTER = {
    "errors_percent": mtick.PercentFormatter(1.0),
    "entity_errors_percent": mtick.PercentFormatter(1.0),
    "entity_errors_percent_lenient": mtick.PercentFormatter(1.0),
    "entity_errors_percent_strict": mtick.PercentFormatter(1.0),
}


def load_xp(path: pl.Path) -> pd.DataFrame:
    df_dict = defaultdict(list)

    with open(path / "config.json") as f:
        config = json.load(f)

    with open(path / "metrics.json") as f:
        data = json.load(f)

        lines = defaultdict(dict)
        for key, metric_dict in data.items():
            metric_name, params = get_params(key)
            params_key = tuple(params[k] for k in XP_PARAMS_KEY["xp_edition"])
            lines[params_key][metric_name] = metric_dict["values"][0]

        for params, metric_dict in lines.items():
            for param_name, param_value in zip(XP_PARAMS_KEY["xp_edition"], params):
                df_dict[param_name].append(param_value)
            for key, value in metric_dict.items():
                df_dict[key].append(value)
            df_dict["hash_len"].append(config["hash_len"])

    df = pd.DataFrame(df_dict)
    return df


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
    parser.add_argument("-o", "--output-file", type=pl.Path, default=None)
    args = parser.parse_args()

    assert len(args.runs) > 0
    df = load_xp(args.runs[0])
    for run in args.runs[1:]:
        run_df = load_xp(run)
        df = pd.concat([df, run_df])
    print(df)

    plt.style.use("science")
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots()
    for i, strat in enumerate(set(df["strategy"])):
        strat_df = df[df["strategy"] == strat]
        strat_df = strat_df.groupby("hash_len", as_index=False)[args.metric].mean()
        hash_len_lst = sorted(set(strat_df["hash_len"]))
        strat_df.loc[:, "x"] = [i + 1 for i, _ in enumerate(hash_len_lst)]
        strat_df.plot(
            ax=ax,
            x="x",
            y=args.metric,
            label=strat,
            marker=MARKERS[i],
            linewidth=1,
            markersize=4,
            alpha=0.75,
        )
        if args.log_scale:
            ax.set_yscale("log")
        if args.metric in METRIC_TO_YFORMATTER:
            ax.yaxis.set_major_formatter(METRIC_TO_YFORMATTER[args.metric])
        ax.set_xticks(list(strat_df["x"]))
        ax.set_xticklabels([str(hash_len) for hash_len in hash_len_lst])
        # add ... for discontinuous regions
        minor_ticks = []
        for i, (hl1, hl2) in enumerate(windowed(hash_len_lst, 2)):
            if hl2 != hl1 + 1:  # type: ignore
                x1 = strat_df.x.loc[i]
                x2 = strat_df.x.loc[i + 1]
                minor_ticks.append(x1 + (x2 - x1) / 2)
        ax.xaxis.set_minor_locator(mtick.FixedLocator(minor_ticks))
        ax.xaxis.set_minor_formatter(mtick.FixedFormatter(["..." for _ in minor_ticks]))
        ax.tick_params(which="minor", length=0)

    ax.grid()
    ax.legend(ncols=2)
    ax.set_xlabel("Hash length")
    ax.set_ylabel(METRIC_TO_YLABEL[args.metric] + " (log)" if args.log_scale else "")

    fig = plt.gcf()
    fig.set_size_inches(4, 2)
    if not args.output_file is None:
        plt.savefig(args.output_file, bbox_inches="tight")
    else:
        plt.show()
