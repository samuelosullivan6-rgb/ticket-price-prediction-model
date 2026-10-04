"""
Chart helpers, so every figure shares one plain style: estimate as a dot,
95% interval as a line, zero as a thin gray reference.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"ols": "#2563eb", "tobit": "#d97706", "neutral": "#9ca3af", "text": "#111827"}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "figure.dpi": 150,
})


def effect_plot(table, path, title, xlabel):
    """
    Dot-and-interval chart for one or two models.
    `table` has columns: label, model, estimate, ci_low, ci_high.
    """

    labels = list(dict.fromkeys(table["label"]))
    models = list(dict.fromkeys(table["model"]))
    offsets = np.linspace(-0.15, 0.15, len(models)) if len(models) > 1 else [0]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(labels) + 1.4))
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)

    for offset, model in zip(offsets, models):
        rows = table[table["model"] == model].set_index("label").reindex(labels)
        y = np.arange(len(labels)) + offset
        color = COLORS.get(model.split()[0].lower(), COLORS["ols"])
        ax.hlines(y, rows["ci_low"], rows["ci_high"], color=color, lw=1.6)
        ax.plot(rows["estimate"], y, "o", color=color, ms=5, label=model)

    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if len(models) > 1:
        ax.legend(frameon=False, loc="lower right", fontsize=8)

    # Long row labels push the axes right, so the title is anchored to the
    # figure's left edge instead of the axes'.
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curve_plot(x, curves, path, title, xlabel, ylabel):
    """Lines with shaded 95% bands. `curves` maps name -> (estimate, low, high)."""

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color=COLORS["neutral"], lw=0.8)

    for name, (estimate, low, high) in curves.items():
        color = COLORS.get(name.split()[0].lower(), COLORS["ols"])
        ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
        ax.plot(x, estimate, color=color, lw=2, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
