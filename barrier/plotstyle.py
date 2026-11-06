"""Shared matplotlib styling (reference palette from the dataviz skill, light mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as tk  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots in fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"RandomWalk": CAT[0], "GBM": CAT[1], "Merton": CAT[2]}
MODEL_LABEL = {"RandomWalk": "Random walk", "GBM": "GBM", "Merton": "Merton jump-diffusion"}


def new_fig(ncols=1, nrows=1, w=5.6, h=4.2, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             facecolor=SURFACE, sharey=sharey, squeeze=False)
    for ax in axes.flat:
        style_ax(ax)
    return fig, axes


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)


def titles(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def legend(ax, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("fontsize", 8.5)
    ax.legend(labelcolor=INK2, **kw)


def pct_axis(axis, decimals=0):
    axis.set_major_formatter(tk.PercentFormatter(1.0, decimals=decimals))


def save(fig, path, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
