"""Plot Top-5 base-ft step and pathway accuracies pre and post fine-tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = "output.png"

# reaction ID, display label, test-step count, pre/post step accuracy (%), pre/post complete-pathway accuracy (%), and FlowER pre-fine-tuning Top-5 step accuracy (%)
REACTIONS = [
    ("simmons_smith", "Simmons-Smith-Furukawa", 3595, 26.0362, 99.5828, 0.0915, 98.6276, 26.48),
    ("cadiot", "Cadiot-Chodkiewicz coupling", 252, 75.7937, 99.2063, 3.1746, 96.8254, 24.60),
    ("larock", "Larock annulation", 3101, 78.7488, 96.3238, 0.3584, 77.4194, 41.63),
    ("decarboxy", "Decarboxylative coupling", 1150, 90.3478, 96.8696, 28.8288, 83.7838, 38.78),
    ("o_allyl", "O-allyl deprotection", 1350, 37.9259, 91.3333, 0.0, 67.0330, 41.56),
    ("tsuji_trost", "Tsuji-Trost allylation", 1521, 44.2472, 94.6746, 0.0, 76.2332, 27.35),
    ("hirao", "Hirao coupling", 1006, 66.5010, 96.8191, 0.0, 74.3590, 37.61),
    ("barluenga", "Barluenga-Valdés coupling", 245, 39.5918, 84.4898, 0.0, 7.6923, 19.18),
    ("ni_negishi", "Nickel Negishi coupling", 507, 30.9665, 92.3077, 0.0, 51.6667, 11.83),
    ("nhk", "Nozaki-Hiyama-Kishi reaction", 124, 12.9032, 74.1935, 0.0, 0.0, 10.58),
    ("rh_hydroformylation", "Rhodium-catalyzed hydroformylation", 97, 12.3711, 88.6598, 0.0, 25.0, 5.15),
    ("fe_kumada", "Iron Kumada coupling", 2502, 14.5084, 97.8018, 0.0, 87.3239, 14.23),
]


def stacked_accuracy_bars(ax, y, before, after, xlabel: str) -> None:
    bar_width = 0.8
    after_increment = [post - pre for pre, post in zip(before, after)]
    ax.barh(
        y,
        before,
        height=bar_width,
        color="#316aac",
        alpha=0.5,
        label="FlowER-OM before fine-tuning",
        align="edge",
    )
    ax.barh(
        y,
        after_increment,
        height=bar_width,
        left=before,
        color="#316aac",
        alpha=1,
        label="FlowER-OM after fine-tuning",
        align="edge",
    )
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=22)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_xlim(0, 100)


def main() -> None:
    labels = [
        rf"$\mathbf{{{index}}}$ {label} ({n_steps:,})"
        for index, (_, label, n_steps, *_accuracies) in enumerate(REACTIONS, start=1)
    ]
    step_before = [row[3] for row in REACTIONS]
    step_after = [row[4] for row in REACTIONS]
    pathway_before = [row[5] for row in REACTIONS]
    pathway_after = [row[6] for row in REACTIONS]
    flower_before = [row[7] for row in REACTIONS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10), sharey=True)
    y = list(range(len(REACTIONS)))
    bin_centres = [index + 0.4 for index in y]

    stacked_accuracy_bars(
        ax1,
        y,
        step_before,
        step_after,
        "Top-5 Step Accuracy (%)",
    )
    ax1.plot(
        flower_before,
        bin_centres,
        color="#d50041",
        linewidth=2,
        linestyle=":",
        marker="o",
        markersize=9,
        label="FlowER before fine-tuning",
        zorder=5,
    )
    ax1.set_yticks(bin_centres)
    ax1.set_yticklabels(labels, fontsize=22)
    ax1.set_ylim(0, len(REACTIONS))
    ax1.invert_yaxis()

    stacked_accuracy_bars(
        ax2,
        y,
        pathway_before,
        pathway_after,
        "Top-5 Pathway Accuracy (%)",
    )

    handles, legend_labels = ax1.get_legend_handles_labels()
    ax2.legend(
        handles,
        legend_labels,
        fontsize=18,
        loc="upper right",
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="#bdbdbd",
    )
    fig.subplots_adjust(left=0.08, wspace=0.05)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=600)
    print(f"Saved {OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()
