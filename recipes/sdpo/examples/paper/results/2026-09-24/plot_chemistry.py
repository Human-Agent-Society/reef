# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Plot the recorded author-reference Chemistry comparison without smoothing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def plot_comparison(result_path: Path, output_prefix: Path) -> None:
    """Render the measured evaluation curves, preserving the training-time boundary."""
    plt.switch_backend("Agg")
    result = json.loads(result_path.read_text())
    budget_hours = 5
    styles = {
        "sdpo": {"color": "#0072B2", "marker": "o", "label": "SDPO"},
        "grpo": {"color": "#D55E00", "marker": "s", "label": "GRPO"},
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "reef-sdpo-chemistry-seed42",
        }
    ):
        figure, axis = plt.subplots(figsize=(8.4, 5.6))
        figure.subplots_adjust(left=0.11, right=0.97, bottom=0.26, top=0.84)
        for method, style in styles.items():
            run = result["methods"][method]
            points = run["evaluations"]
            eligible = [point for point in points if point["training_seconds"] <= budget_hours * 3600]
            after_budget = [point for point in points if point["training_seconds"] > budget_hours * 3600]
            best = max(eligible, key=lambda point: point["avg_at_16"])
            recorded_best = run["best_within_budget"]["18000"]
            if best["step"] != recorded_best["step"] or not math.isclose(
                best["avg_at_16"], recorded_best["avg_at_16"], abs_tol=1e-12
            ):
                raise ValueError(f"The recorded five-hour maximum is inconsistent for {method}")
            label = f"{style['label']} (minibatch {run['minibatch']}, LR {run['learning_rate']:.0e})"
            axis.plot(
                [point["training_seconds"] / 3600 for point in eligible],
                [point["avg_at_16"] * 100 for point in eligible],
                color=style["color"],
                marker=style["marker"],
                markersize=3.3,
                linewidth=1.8,
                label=label,
            )
            if after_budget:
                continuation = [eligible[-1], *after_budget]
                axis.plot(
                    [point["training_seconds"] / 3600 for point in continuation],
                    [point["avg_at_16"] * 100 for point in continuation],
                    color=style["color"],
                    linewidth=1.3,
                    linestyle="--",
                )
                axis.plot(
                    [point["training_seconds"] / 3600 for point in after_budget],
                    [point["avg_at_16"] * 100 for point in after_budget],
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markersize=5,
                    linestyle="none",
                )
            axis.annotate(
                f"{best['avg_at_16'] * 100:.2f}%",
                xy=(best["training_seconds"] / 3600, best["avg_at_16"] * 100),
                xytext=(-20, 15),
                textcoords="offset points",
                ha="center",
                color=style["color"],
                arrowprops={"arrowstyle": "-", "color": style["color"], "linewidth": 0.8},
            )
        axis.axvspan(budget_hours, 5.18, color="0.92", zorder=0)
        axis.axvline(budget_hours, color="0.45", linestyle=":", linewidth=1)
        axis.text(4.96, 87, "5 h budget", ha="right", va="top", color="0.35", fontsize=9)
        axis.set(
            xlim=(0, 5.18), ylim=(0, 90), xlabel="Cumulative training time (hours)", ylabel="Test accuracy (avg@16, %)"
        )
        axis.set_xticks(range(6))
        axis.set_yticks(range(0, 81, 20))
        axis.grid(axis="y", color="0.86", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", frameon=False, fontsize=9)
        figure.suptitle("SDPO vs. GRPO on Chemistry", y=0.96, fontsize=15)
        figure.text(
            0.5, 0.9, "OLMo-3-7B-Instruct | 4 x H100 80GB | seed 42 | author implementation", ha="center", fontsize=10
        )
        figure.text(
            0.11,
            0.15,
            "Both methods: 32 questions x 8 responses per rollout; 210 test questions x 16 samples.",
            fontsize=9,
        )
        figure.text(
            0.11, 0.11, "Each marker is an evaluation; lines connect measurements without smoothing.", fontsize=9
        )
        figure.text(
            0.11,
            0.07,
            "Hollow markers exceed 5 h and are excluded from peaks; timing excludes initialization and evaluation.",
            fontsize=8,
        )
        figure.text(
            0.11,
            0.03,
            "Single-seed default-configuration comparison; stopping at the time budget does not establish convergence.",
            fontsize=8,
        )
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_prefix.with_suffix(".png"), dpi=240)
        svg_path = output_prefix.with_suffix(".svg")
        figure.savefig(svg_path, metadata={"Date": None})
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        plt.close(figure)


def main() -> None:
    """Render the versioned result next to this script, or supplied paths."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=directory / "author-reference-chemistry-seed42.json")
    parser.add_argument("--output-prefix", type=Path, default=directory / "chemistry-seed42-learning-curves")
    arguments = parser.parse_args()
    plot_comparison(arguments.result, arguments.output_prefix)


if __name__ == "__main__":
    main()
