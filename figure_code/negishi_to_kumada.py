"""Negishi-to-Kumada transfer figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

FIGURE_DIR = Path(__file__).resolve().parent
COLOR = "#316aac"
BASE_ALPHA = 0.5
BAR_WIDTH = 0.30
BAR_SPACING = 0.05
FONT_SIZE_LABEL = 22
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICK = 18
FONT_SIZE_TITLE = 22


# Top-5 step accuracy in percent, ordered Negishi, Kumada.
BASE_ACCURACY = [16.7653, 18.49]
FINE_TUNED_ACCURACY = [91.9132, 91.0753]
REACTION_LABELS = ["Negishi\n(Used for\nfine-tuning)", "Kumada\n(Omitted from\nfine-tuning)"]


def main(title: str = "Negishi-to-Kumada transfer") -> None:
    spacing_scale = 1.3
    panel_width, panel_height = 4.3 * spacing_scale, 3.78
    height, left, bottom = 7.2, 1.17, 2.0
    width = left + panel_width + 0.25
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, bottom / height,
                       panel_width / width, panel_height / height))
    x = np.arange(len(REACTION_LABELS)) * spacing_scale
    offset = BAR_WIDTH / 2 + BAR_SPACING
    ax.bar(x - offset, BASE_ACCURACY, BAR_WIDTH, color=COLOR, alpha=BASE_ALPHA)
    ax.bar(x + offset, FINE_TUNED_ACCURACY, BAR_WIDTH, color=COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(REACTION_LABELS, fontsize=FONT_SIZE_TICK)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.set_xlim(-0.70 * spacing_scale, 1.70 * spacing_scale)
    ax.set_ylim(0, 108)
    for container in ax.containers:
        ax.bar_label(container,
                     labels=[f"{bar.get_height():.1f}%" for bar in container],
                     padding=4, fontsize=14)
    ax.set_ylabel("Top-5 Step Accuracy (%)", fontsize=FONT_SIZE_LABEL)
    #ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=20)
    handles = [
        Patch(facecolor=COLOR, alpha=BASE_ALPHA, label="Base model"),
        Patch(facecolor=COLOR, label="Fine-tuned on nickel Negishi"),
    ]
    fig.legend(handles=handles, loc="center",
               bbox_to_anchor=(0.5, 0.68 / height), ncol=2,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.5,
               fontsize=FONT_SIZE_LEGEND, frameon=False)
    fig.canvas.draw()
    box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    assert np.allclose((box.width, box.height), (panel_width, panel_height))
    for extension in (["png"]):
        fig.savefig(FIGURE_DIR / f"transfer.{extension}", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    main()
