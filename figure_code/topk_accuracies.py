"""Plot user-specified top-k step and pathway accuracies."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transfer import (FIGURE_HEIGHT, AXES_BOTTOM, AXES_TOP, FONT_SIZE_LABEL,
                      FONT_SIZE_TICK, FONT_SIZE_LEGEND)

ROOT = Path(__file__).resolve().parents[1]
COLOR = "#316aac"


# Enter fractions (0–1) in order from top-1 to top-5.
STEP_ACCURACY = [0.87473273, 0.95261251, 0.97032347, 0.97528815, 0.97630785]
PATHWAY_ACCURACY = [0.60477364, 0.84029583, 0.88897355, 0.90221874, 0.90481847]
STEP_CEILING = [0.9085852632920264, 0.9786594883883916, 0.9936892739957313, 0.9982606479564207, 0.999503042273263]
PATHWAY_CEILING = [0.6530479605558046, 0.9241371582250112, 0.9750784401613626, 0.9922456297624384, 0.9976019722097714]


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": 16,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    fig, axes = plt.subplots(1, 2, figsize=(12, FIGURE_HEIGHT))
    k_values = range(1, 6)
    for ax, metric, accuracy, ceiling, ticks, limits in (
        (axes[0], "step", STEP_ACCURACY, STEP_CEILING, [84, 88, 92, 96, 100], (84, 101)),
        (axes[1], "pathway", PATHWAY_ACCURACY, PATHWAY_CEILING, [50, 60, 70, 80, 90, 100], (50, 102)),
    ):
        ax.plot(k_values, [100 * value for value in accuracy],
                color=COLOR, marker="o", linewidth=2, label="FlowER-OM")
        ax.plot(k_values, [100 * value for value in ceiling],
                color="black", linestyle=":", linewidth=2,
                label="Maximum possible accuracy")
        ax.set_xlabel("$k$", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(f"Top-$k$ {metric.capitalize()} Accuracy (%)",
                      fontsize=FONT_SIZE_LABEL)
        ax.set_xticks(k_values)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.set_ylim(*limits)
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=FONT_SIZE_LEGEND, frameon=False,
               bbox_to_anchor=(.53, .95))
    fig.subplots_adjust(bottom=AXES_BOTTOM, top=AXES_TOP,
                        left=.10, right=.97, wspace=.30)
    stem = ROOT / ""
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
