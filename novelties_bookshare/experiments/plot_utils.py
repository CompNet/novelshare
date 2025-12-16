MARKERS = ["X", "p", "*", "D", "^", "v", "1", "o", "s"]


def by_strat(strat: str) -> int:
    """Utility function to sort strategies in order"""
    strats = ["naive", "case", "split", "mlm", "propagate", "pipe"]
    try:
        return strats.index(strat)
    except ValueError:
        return -1


# from
# https://github.com/garrettj403/SciencePlots/blob/master/src/scienceplots/styles/science.mplstyle
STRAT_COLOR_HINTS = {
    "naive": "#0C5DA5",
    "case": "#00B945",
    "split": "#FF9500",
    "mlm": "#FF2C00",
    "propagate": "#845B97",
    "pipe": "#474747",
}
