import argparse, json, re, os, math
from collections import defaultdict
import functools as ft
import pathlib as pl
import pandas as pd
import matplotlib.pyplot as plt


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
        return list(
            range(config["min_errors"], config["max_errors"], config["errors_step"])
        )
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

    # one subplot per "noise"
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
            for book in set(df["book"]):
                ax_df[ax_df["book"] == book].plot(  # type: ignore
                    ax=ax, x="steps", y="values", title=noise, label=book
                )
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
        plt.tight_layout()
        out_path = args.output_dir / f"perbook_{strat}.png"
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
            for strat in set(df["strat"]):
                ax_df[ax_df["strat"] == strat].plot(  # type: ignore
                    ax=ax, x="steps", y="values", title=noise, label=strat
                )
            ax.set_ylabel(METRIC2PRETTY[args.metric])
            ax.set_xlabel(info.get(f"{noise}.errors_unit", "steps"))
        plt.tight_layout()
        out_path = args.output_dir / f"perstrat_{book}.png"
        print(f"saving {out_path}")
        plt.savefig(out_path)
