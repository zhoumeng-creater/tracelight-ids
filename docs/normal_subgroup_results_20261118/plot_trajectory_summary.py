"""Static figures from existing event audit only; no models or score recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot(input_path, output_path):
    audit = json.loads(input_path.read_text(encoding="utf-8"))
    curves = audit["trajectories"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 11, "axes.titlesize": 12, "savefig.facecolor": "white"})
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.7), sharex=True, sharey="row")
    colors = {"transport_target": "#0072B2", "identity_target": "#D55E00"}
    for column, seed in enumerate((17, 29, 43)):
        baseline = curves[f"transport_target_s{seed}"]["baseline"]
        axes[0, column].set_title(f"Seed {seed}", fontweight="bold", pad=11)
        for row, metric in enumerate(("H", "H99")):
            axis = axes[row, column]
            base = baseline["H"]["value"] if metric == "H" else baseline["H_q"]["0.99"]["value"]
            axis.axhline(base, color="#555555", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            for arm in ("transport_target", "identity_target"):
                trajectory = curves[f"{arm}_s{seed}"]
                assert trajectory["execution"]["selected_step"] == 0
                assert not trajectory["eligible_steps"]
                events = trajectory["events"]
                x = [0] + [event["step"] for event in events]
                y = [base] + [(event["H"]["value"] if metric == "H"
                              else event["H_q"]["0.99"]["value"]) for event in events]
                axis.plot(x, y, color=colors[arm], linewidth=1.7,
                          marker="o" if arm == "transport_target" else "s",
                          markersize=3.5, markerfacecolor="white", markeredgewidth=.9,
                          zorder=3 if arm == "transport_target" else 2)
            axis.grid(axis="y", color="#E2E2E2", linewidth=.7)
            axis.yaxis.set_major_locator(MaxNLocator(5))
            axis.set_xlim(-35, 2265)
            axis.set_xticks([0, 556, 1112, 1668, 2224])
            axis.tick_params(axis="both", length=3, color="#777777")
            if row == 1:
                axis.set_xlabel("Optimizer updates")
    axes[0, 0].set_ylabel(r"Pooled $H=\max(H_{90},H_{95},H_{99})$")
    axes[1, 0].set_ylabel(r"Pooled $H_{99}$ (top 1% excess)")
    axes[1, 0].annotate("Above baseline in first 12 checks",
                        xy=(834, curves["transport_target_s17"]["events"][5]["H_q"]["0.99"]["value"]),
                        xytext=(420, .195), fontsize=9, color="#0072B2",
                        arrowprops={"arrowstyle": "-", "color": "#0072B2", "linewidth": .8})
    legend = [Line2D([0], [0], color=colors["transport_target"], marker="o", markersize=4,
                     markerfacecolor="white", linewidth=1.7, label="Transport (all checkpoints rejected)"),
              Line2D([0], [0], color=colors["identity_target"], marker="s", markersize=4,
                     markerfacecolor="white", linewidth=1.7, label="Identity control"),
              Line2D([0], [0], color="#555555", linestyle=(0, (4, 3)), linewidth=1.2, label="Same-seed R30 baseline")]
    figure.suptitle("Normal-tail trajectories: overall H improvement does not guarantee top-1% protection",
                    fontsize=14, fontweight="bold", y=.976)
    figure.text(.5, .934,
                "All 48 transport checkpoints were rejected; all six final artifacts fell back to R30 (step 0).",
                ha="center", fontsize=10.5, color="#333333")
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .905), ncol=3,
                  frameon=False, fontsize=10, handlelength=2.4, columnspacing=2.2)
    figure.text(.07, .024,
                "Recorded CPU validation metrics only. Lower is better. Markers: 139-update intervals; step 0 is the baseline.\n"
                "H99 measures subgroup enrichment within the top 1% of pooled normal scores, not deployment FPR.",
                ha="left", va="bottom", fontsize=9, color="#444444", linespacing=1.4)
    figure.subplots_adjust(left=.075, right=.985, top=.815, bottom=.14, hspace=.26, wspace=.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("training_audit.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trajectory_summary.png"))
    arguments = parser.parse_args()
    plot(arguments.input.resolve(), arguments.output.resolve())
