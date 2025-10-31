import argparse, json, re, os, math
from collections import defaultdict
import functools as ft
import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt


@ft.lru_cache
def get_params(metric_key: str) -> dict[str, str]:
    # form of each metric
    # b=book.s=strat.n=noise.recovered_tokens_proportion
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
    with open(args.run / "metrics.json") as f:
        metrics = json.load(f)
    # some metrics are not ordered in terms of steps
    for v in metrics.values():
        sort_idx = sorted(list(range(len(v["steps"]))), key=lambda i: v["steps"][i])
        for k in v.keys():
            v[k] = [v[k][i] for i in sort_idx]
    return metrics


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
        ocr_run_metrics = load_metrics(args.ocr_run)
        metrics = {**ocr_run_metrics, **metrics}
    else:
        print("OCR run not specified. Not plotting OCR results.")

    # construct df
    df_dict = defaultdict(list)
    for k, v in metrics.items():
        params = get_params(k)
        if params["metric"] != args.metric:
            continue
        for step, value in zip(v["steps"], v["values"]):
            df_dict["book"].append(params["book"])
            df_dict["strat"].append(params["strat"])
            df_dict["noise"].append(params["noise"])
            df_dict["steps"].append(step)
            df_dict["values"].append(value)
    df = pd.DataFrame(df_dict)
    print(df)

    # plot
    os.makedirs(args.output_dir, exist_ok=True)

    # one subplot per "noise"
    # pick "book" to split curves
    for strat in set(df["strat"]):
        noises = list(set(df["noise"]))
        fig, axs = plt.subplots(math.ceil(len(noises) / 2), 2, figsize=(12, 8))
        fig.suptitle(strat)
        for i, noise in enumerate(noises):
            ax = axs[i // 2][i % 2]
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax_df = df[(df["strat"] == strat) & (df["noise"] == noise)]
            for book in set(df["book"]):
                ax_df[ax_df["book"] == book].plot(  # type: ignore
                    ax=ax, x="steps", y="values", title=noise, label=book
                )
        plt.tight_layout()
        out_path = args.output_dir / f"perbook_{strat}.png"
        print(f"saving {out_path}")
        plt.savefig(out_path)
        plt.close("all")

    # pick "strat" to split curves
    for book in set(df["book"]):
        noises = list(set(df["noise"]))
        fig, axs = plt.subplots(math.ceil(len(noises) / 2), 2, figsize=(12, 8))
        fig.suptitle(book)
        for i, noise in enumerate(noises):
            ax = axs[i // 2][i % 2]
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax_df = df[(df["book"] == book) & (df["noise"] == noise)]
            for strat in set(df["strat"]):
                ax_df[ax_df["strat"] == strat].plot(  # type: ignore
                    ax=ax, x="steps", y="values", title=noise, label=strat
                )
        plt.tight_layout()
        out_path = args.output_dir / f"perstrat_{book}.png"
        print(f"saving {out_path}")
        plt.savefig(out_path)
