MARKERS = ["X", "p", "*", "D", "^", "v", "1", "o", "s"]

MARKER_PAIRS = [("P", "X"), ("s", "D"), ("^", "v"), ("<", ">")]


def by_strat(strat: str) -> int:
    """Utility function to sort strategies in order"""
    # NOTE: split is deprecated for "retokenize"
    strats = ["naive", "case", "retokenize", "split", "mlm", "propagate", "pipe"]
    try:
        return strats.index(strat)
    except ValueError:
        return -1


# NOTE: split is deprecated for "retokenize"
STRAT_MARKERS_HINT = {
    "naive": ".",
    "case": "v",
    "retokenize": "*",
    "split": "*",
    "mlm": "X",
    "propagate": "d",
    "pipe": "h",
}

# from
# https://github.com/garrettj403/SciencePlots/blob/master/src/scienceplots/styles/science.mplstyle
COLORS = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747"]

COLOR_PAIRS = [
    ("#0C5DA5", "#0C89A5"),
    ("#FF2C00", "#FF9500"),
    ("#00B945", "#00b991"),
]

STRAT_COLOR_HINTS = {
    "naive": "#0C5DA5",
    "case": "#00B945",
    "split": "#FF9500",  # deprecated for 'retokenize'
    "retokenize": "#FF9500",
    "mlm": "#FF2C00",
    "propagate": "#845B97",
    "pipe": "#474747",
}

EDITION_COLOR_HINTS = {
    "MD-1851-UK": "#FF2C00",
    "MD-1988": "#00B945",
    "F-1823": "#00B945",
    "F-1831": "#FF2C00",
    "PP-1817": "#00B945",
    "PP-1894": "#FF2C00",
}
