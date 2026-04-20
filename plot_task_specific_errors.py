import math, argparse, json, re
import functools as ft
from collections import defaultdict
import pathlib as pl
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from plot_xp_synthetic_errors import (
    get_params,
    load_metrics,
    load_config,
    load_info,
    get_steps,
    format_ocr_xtick,
    METRIC2PRETTY,
    METRIC_TO_YFORMATTER,
)
from novelties_bookshare.experiments.plot_utils import (
    MARKER_PAIRS,
    STRAT_MARKERS_HINT,
    COLOR_PAIRS,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--runs", type=pl.Path, nargs="+", help="Runs of xp_synthetic_errors.py"
    )
    parser.add_argument(
        "-c",
        "--ocr-runs",
        type=pl.Path,
        nargs="+",
        help="Runs of xp_synthetic_ocr_errors.py",
    )
    parser.add_argument(
        "-t",
        "--task-metrics",
        type=str,
        nargs="+",
        help="one of: 'entity_errors_nb', 'entity_errors_percent', 'coref_mention_errors_nb', 'coref_mention_errors_percent'",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        help="one of: 'errors_nb', 'duration_s', 'errors_percent'",
    )
    parser.add_argument("-l", "--labels", type=str, nargs="+")
    parser.add_argument("-o", "--output-file", type=pl.Path, default=None)
    args = parser.parse_args()

    df_dict = defaultdict(list)
    for run, ocr_run, label in zip(args.runs, args.ocr_runs, args.labels):
        metrics = load_metrics(run)
        ocr_metrics = load_metrics(ocr_run)
        metrics = {**ocr_metrics, **metrics}

        config = load_config(run)
        ocr_config = load_config(ocr_run)
        config = {**config, **ocr_config}

        info = load_info(run)
        ocr_info = load_info(ocr_run)
        info = {**ocr_info, **info}

        for k, v in metrics.items():
            params = get_params(k)
            steps = get_steps(params["noise"], config)
            for step, value in zip(steps, v["values"]):
                df_dict["book"].append(params["book"])
                df_dict["strat"].append(params["strat"])
                df_dict["noise"].append(params["noise"])
                df_dict["steps"].append(step)
                df_dict["values"].append(value)
                df_dict["label"].append(label)
                df_dict["metric"].append(params["metric"])
        df = pd.DataFrame(df_dict)

    fig, axs = (None, None)
    plt.style.use("science")
    cols_nb = 3
    plt.rcParams.update({"font.size": 16})

    for run_i, (run, ocr_run, metric, task_metric, label) in enumerate(
        zip(args.runs, args.ocr_runs, args.metrics, args.task_metrics, args.labels)
    ):
        noises = sorted(set(df["noise"]))
        if fig is None and axs is None:
            fig, axs = plt.subplots(
                math.ceil(len(noises) / cols_nb), cols_nb, figsize=(16, 8)
            )

        # we average per "book". Note that we originally experimented
        # only with novels so this denomination made sense, but for
        # others datasets such as CoNLL-2003, a "book" is rather a
        # "document".
        run_df = df.copy()
        run_df = run_df[df.label == label]
        # groupby can't handle a mix of floats and tuples
        run_df["steps"] = run_df["steps"].astype(str)
        run_df = run_df.groupby(
            ["strat", "noise", "steps", "metric"], as_index=False
        ).agg({"values": "mean", "strat": "first", "noise": "first", "steps": "first"})

        for i, noise in enumerate(noises):
            ax = axs[i // cols_nb][i % cols_nb]
            ax_df = run_df[(run_df["noise"] == noise) & (run_df["strat"] == "pipe")]
            ax_df[ax_df.metric == task_metric].plot(
                ax=ax,
                x="steps",
                y="values",
                title="\\texttt{{{0}}}".format(noise),
                label=f"{label} (entities)",
                marker=MARKER_PAIRS[run_i][0],
                alpha=0.75,
                c=COLOR_PAIRS[run_i][0],
            )
            ax_df[ax_df.metric == metric].plot(
                ax=ax,
                x="steps",
                y="values",
                title="\\texttt{{{0}}}".format(noise),
                label=f"{label} (all tokens)",
                marker=MARKER_PAIRS[run_i][1],
                alpha=0.75,
                c=COLOR_PAIRS[run_i][1],
            )
            ax.set_ylabel(METRIC2PRETTY[metric])
            if metric in METRIC_TO_YFORMATTER:
                ax.yaxis.set_major_formatter(METRIC_TO_YFORMATTER[metric])
            ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
            ax.grid()
            if noise == "ocr_scramble":
                xticks = ax.get_xticklabels()
                ax.set_xticklabels(
                    [format_ocr_xtick(xtick.get_text()) for xtick in xticks]
                )

    plt.tight_layout()
    if args.output_file is None:
        plt.show()
    else:
        print(f"saving {args.output_file}")
        plt.savefig(args.output_file)
