import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=Path, nargs="*")
    parser.add_argument("-s", "--style", type=str, default="science")
    parser.add_argument("-o", "--output-file", type=Path, default=None)
    args = parser.parse_args()

    plt.style.use(args.style)
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots()
    for i, run in enumerate(args.runs):
        with open(run / "run.json") as f:
            run_details = json.load(f)
        model_name = run_details["meta"]["config_updates"]["model_name"]
        with open(run / "metrics.json") as f:
            data = json.load(f)
        # alignment.errors_percent is common to all xp, we just need
        # to plot it once
        if i == 0:
            ax.plot(
                data["alignment.errors_percent"]["steps"],
                data["alignment.errors_percent"]["values"],
                label="novelshare",
                marker=".",
            )
        ax.plot(
            data["model.errors_percent"]["steps"],
            data["model.errors_percent"]["values"],
            label=model_name,
            marker=".",
        )
    ax.legend()
    ax.set_xlabel("hash size")
    ax.set_ylabel("percentage of token errors")
    ax.grid(True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if args.output_file is None:
        plt.show()
    else:
        plt.savefig(args.output_file)
