"""Step holdout figure, optionally with a marked table."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGURE_DIR = Path(__file__).resolve().parent
COLOR = "#316aac"
BASE_ALPHA = 0.5
BAR_WIDTH = 0.30
BAR_SPACING = 0.03
FONT_SIZE_LABEL = 22
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICK = 18


# Top-5 step accuracy in percent, ordered from the top bar pair down.
# RE to Ni(0), product-forming RE, all transmetalation, all RE.
BASE_ACCURACY = [0.0, 15.0, 34.93150684931507, 9.473684210526315]
FINE_TUNED_ACCURACY = [40.0, 96.6667, 54.794520547945204, 40.0]


def draw_step_table(ax, y, limits, tested_colors) -> None:
    # Rows follow the bar pairs; columns follow steps 1–5 in the reference.
    states = ("ITIIH", "IHIIT", "THITH", "HTIHT")
    gray = "#555555"
    for step in (1, 3, 5):
        ax.axvspan(step - 0.5, step + 0.5, color="#f2f2f2", linewidth=0)
    for row, (center, steps) in enumerate(zip(y, states)):
        for step, state in enumerate(steps, start=1):
            ax.plot(step, center, linestyle="none", markersize=16,
                    marker="s" if state == "T" else "o",
                    color=tested_colors[row] if state == "T" else gray,
                    markerfacecolor="none" if state == "H" else "auto",
                    markeredgewidth=1.5 if state == "H" else 0)
    ax.set(xlim=(0.5, 5.5), ylim=limits, yticks=[])
    ax.set_xticks(range(1, 6), labels=[str(step) for step in range(1, 6)], fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2, labelsize=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=gray, linestyle="none", label="In fine-tuning data"),
            Line2D([], [], marker="o", color=gray, markerfacecolor="none",
                   linestyle="none", label="Held out"),
            Line2D([], [], marker="s", color=gray, linestyle="none", label="Held out & tested"),
        ],
        loc="upper left", bbox_to_anchor=(0, -0.012), borderaxespad=0,
        frameon=False, fontsize=15, markerscale=1.7, labelspacing=0.18,
        handlelength=1.0, handletextpad=0.45, borderpad=0.25,
    )


def main(with_table: bool = False) -> None:
    table_width = 2.65 if with_table else 0
    plot_width = 4.85 if with_table else 5.5
    width = 0.5 + plot_width + table_width
    pair_pitch = 0.72 if with_table else 0.80
    y = np.arange(4) * pair_pitch
    lower, upper = -0.4, y[-1] + 0.4
    # Match the compact reaction panel's 4.3 inches per 2.4 category units.
    axes_height = (upper - lower) * (4.3 / 2.4)
    bottom = 0.89 if with_table else 0.75
    height = axes_height + bottom + (0.36 if with_table else 0.23)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(((0.25 + table_width) / width, bottom / height,
                       plot_width / width, axes_height / height))
    offset = BAR_WIDTH / 2 + (0.01 if with_table else BAR_SPACING)
    blue_base = Patch(facecolor=COLOR, alpha=BASE_ALPHA).get_facecolor()
    red_base, red_tuned = "#ff4d83", "#d50041"
    if with_table:
        table = fig.add_axes((0.20 / width, bottom / height,
                              (table_width - 0.10) / width, axes_height / height))
        draw_step_table(table, y, (upper, lower), (COLOR, COLOR, red_tuned, red_tuned))
    for positions, values, colors in (
        (y - offset, BASE_ACCURACY, [blue_base] * 2 + [red_base] * 2),
        (y + offset, FINE_TUNED_ACCURACY, [COLOR] * 2 + [red_tuned] * 2),
    ):
        bars = ax.barh(positions, values, height=BAR_WIDTH, color=colors)
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=14)

    ax.set_yticks([])
    ax.set_ylim(upper, lower)
    ax.set_xticks(range(0, 101, 25 if with_table else 20))
    ax.set_xlim(0, 114 if with_table else 112)
    ax.set_xlabel("Top-5 Step Accuracy (%)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    ax.legend(
        handles=[(Patch(facecolor=blue_base), Patch(facecolor=red_base)),
                 (Patch(facecolor=COLOR), Patch(facecolor=red_tuned))],
        labels=["Base model", "Fine-tuned"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="upper right", handlelength=3,
        fontsize=FONT_SIZE_LEGEND, frameon=False,
    )
    for extension in ("png", "pdf", "svg"):
        suffix = "_table" if with_table else ""
        output = FIGURE_DIR / f"step_holdout{suffix}.{extension}"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-table", action="store_true")
    main(with_table=parser.parse_args().with_table)
