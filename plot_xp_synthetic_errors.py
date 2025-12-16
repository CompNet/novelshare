import argparse, json, re, os, math, ast
from collections import defaultdict
import functools as ft
import pathlib as pl
import pandas as pd
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

MARKERS = ["X", "p", "*", "D", "^", "v", "1", "o", "s"]


@ft.lru_cache
def get_params(metric_key: str) -> dict[str, str]:
    # form of each metric
    # b=book.s=strat.n=noise.metric_name
    # or
    # b=book.s=strat.n=noise.w=target_wer.c=target_cer.metric_name
    m = re.match(
        r"b=([^\.]+)\.s=([^\.]+)\.n=([^\.]+)\.w=([^c]+)\.c=([^a-z]+)\.(.*)", metric_key
    )
    if not m is None:
        book, strat, noise, wer, cer, metric = m.groups()
    else:
        m = re.match(r"b=([^\.]+)\.s=([^\.]+)\.n=([^\.]+)\.(.*)", metric_key)
        if m is None:
            return {}
        book, strat, noise, metric = m.groups()
    return {
        "book": book,
        "strat": strat,
        "noise": noise,
        "metric": metric,
    }


def load_metrics(run: pl.Path) -> dict:
    # load metrics
    with open(run / "metrics.json") as f:
        metrics = json.load(f)
    # some metrics are not ordered in terms of steps
    for v in metrics.values():
        sort_idx = sorted(list(range(len(v["steps"]))), key=lambda i: v["steps"][i])
        for k in v.keys():
            v[k] = [v[k][i] for i in sort_idx]
    return metrics


def load_config(run: pl.Path) -> dict:
    with open(run / "config.json") as f:
        config = json.load(f)
    return config


def load_info(run: pl.Path) -> dict:
    with open(run / "info.json") as f:
        config = json.load(f)
    return config


def get_steps(noise: str, config: dict) -> list:
    if noise in {"add", "delete", "substitute", "token_merge", "token_split"}:
        return [
            float(step)
            for step in np.arange(
                config["min_error_ratio"],
                config["max_error_ratio"],
                config["error_ratio_step"],
            )
        ]
    elif noise == "ocr_scramble":
        return list(zip(config["wer_grid"], config["cer_grid"]))
    else:
        raise ValueError(noise)


METRIC2PRETTY = {
    "errors_nb": "Number of errors",
    "duration_s": "Duration (s)",
    "errors_percent": "Percentage of errors",
    "entity_errors_nb": "Number of entity errors",
    "entity_errors_percent": "Percentage of entity errors",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run", type=pl.Path, help="Run of xp_synthetic_errors.py"
    )
    parser.add_argument(
        "-c",
        "--ocr-run",
        type=pl.Path,
        default=None,
        help="Run of xp_synthetic_ocr_errors.py",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        help="one of: 'errors_nb', 'duration_s', 'errors_percent', 'entity_errors_nb', 'entity_errors_percent'",
    )
    parser.add_argument("-o", "--output-dir", type=pl.Path)
    args = parser.parse_args()

    metrics = load_metrics(args.run)
    if args.ocr_run:
        ocr_metrics = load_metrics(args.ocr_run)
        metrics = {**ocr_metrics, **metrics}
    else:
        print("OCR run not specified. Not plotting OCR results.")

    config = load_config(args.run)
    if args.ocr_run:
        ocr_config = load_config(args.ocr_run)
        config = {**ocr_config, **config}

    info = load_info(args.run)
    if args.ocr_run:
        ocr_info = load_info(args.ocr_run)
        info = {**ocr_info, **info}

    # construct df
    df_dict = defaultdict(list)
    for k, v in metrics.items():
        params = get_params(k)
        if params["metric"] != args.metric:
            continue
        steps = get_steps(params["noise"], config)
        for step, value in zip(steps, v["values"]):
            df_dict["book"].append(params["book"])
            df_dict["strat"].append(params["strat"])
            df_dict["noise"].append(params["noise"])
            df_dict["steps"].append(step)
            df_dict["values"].append(value)
    df = pd.DataFrame(df_dict)
    print(df)

    # plot
    os.makedirs(args.output_dir, exist_ok=True)
    cols_nb = 3
    plt.style.use("science")
    plt.rcParams.update({"font.size": 16})

    # # one subplot per "noise"
    # pick "book" to split curves
    for strat in set(df["strat"]):
        noises = list(set(df["noise"]))
        fig, axs = plt.subplots(
            math.ceil(len(noises) / cols_nb), cols_nb, figsize=(16, 6)
        )
        fig.suptitle(strat)
        for i, noise in enumerate(noises):
            ax = axs[i // cols_nb][i % cols_nb]
            ax_df = df[(df["strat"] == strat) & (df["noise"] == noise)]
            for j, book in enumerate(set(df["book"])):
                ax_df[ax_df["book"] == book].plot(
                    ax=ax,
                    x="steps",
                    y="values",
                    title=noise,
                    label=book,
                    marker=MARKERS[j],
                )
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
            ax.grid()
        plt.tight_layout()
        out_path = args.output_dir / f"perbook_{strat}.pdf"
        print(f"saving {out_path}")
        plt.savefig(out_path)
        plt.close("all")

    # pick "strat" to split curves
    for book in set(df["book"]):
        noises = list(set(df["noise"]))
        fig, axs = plt.subplots(
            math.ceil(len(noises) / cols_nb), cols_nb, figsize=(16, 8)
        )
        fig.suptitle(book)
        for i, noise in enumerate(noises):
            ax = axs[i // cols_nb][i % cols_nb]
            ax_df = df[(df["book"] == book) & (df["noise"] == noise)]
            for j, strat in enumerate(set(df["strat"])):
                ax_df[ax_df["strat"] == strat].plot(
                    ax=ax,
                    x="steps",
                    y="values",
                    title=noise,
                    label=strat,
                    marker=MARKERS[j],
                    markersize=12 - j,
                    alpha=0.75,
                )
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
            ax.grid()
        plt.tight_layout()
        out_path = args.output_dir / f"perstrat_{book}.pdf"
        print(f"saving {out_path}")
        plt.savefig(out_path)
    plt.close("all")

    # pick "strat" to split curves and average over books
    noises = list(set(df["noise"]))
    fig, axs = plt.subplots(math.ceil(len(noises) / cols_nb), cols_nb, figsize=(16, 8))
    df_book_avg = df.copy()
    # groupby can't handle a mix of floats and tuples
    df_book_avg["steps"] = df_book_avg["steps"].astype(str)
    df_book_avg = df_book_avg.groupby(["strat", "noise", "steps"], as_index=False).mean(
        "values"
    )
    df_book_avg["steps"] = df_book_avg["steps"].apply(ast.literal_eval)
    for i, noise in enumerate(noises):
        ax = axs[i // cols_nb][i % cols_nb]
        ax_df = df_book_avg[df_book_avg["noise"] == noise]
        for j, strat in enumerate(set(ax_df["strat"])):
            ax_strat_df = ax_df[ax_df["strat"] == strat]
            ax_strat_df.plot(
                ax=ax,
                x="steps",
                y="values",
                title=noise,
                label=strat,
                marker=MARKERS[j],
                markersize=12 - j,
                alpha=0.75,
            )
        ax.set_ylabel(METRIC2PRETTY[args.metric])
        ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
        ax.grid()
    plt.tight_layout()
    out_path = args.output_dir / f"perstrat_average.pdf"
    print(f"saving {out_path}")
    plt.savefig(out_path)
