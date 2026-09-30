"""Plot the complete AIME curve and assess the predeclared final-checkpoint target."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_scores(path: Path) -> tuple[dict[tuple[str, int], bool], str]:
    scores = {}
    releases = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["question_id"], row["seed"])
        if key in scores:
            raise ValueError(f"Duplicate question/seed in {path}: {key}")
        if row.get("evaluation") is not True or not isinstance(row.get("correct"), bool):
            raise ValueError(f"Expected scored held-out evaluation in {path}")
        scores[key] = row["correct"]
        releases.add(row.get("release_id"))
    if not scores or len(releases) != 1 or None in releases:
        raise ValueError(f"Expected one identifiable release and nonempty predictions in {path}")
    return scores, releases.pop()


def paired_interval(baseline: dict, final: dict, *, draws: int = 10000) -> list[float]:
    """Resample questions, keeping each question's repetitions and checkpoints paired."""
    if baseline.keys() != final.keys():
        raise ValueError("Evaluation question/seed sets differ between checkpoints")
    differences = defaultdict(list)
    for key, correct in baseline.items():
        differences[key[0]].append(int(final[key]) - int(correct))
    values = [sum(items) / len(items) for _, items in sorted(differences.items())]
    rng = random.Random(0)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return [samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]]


def analyze(
    results: Path,
    *,
    final_step: int = 200,
    eval_every: int = 20,
    questions: int = 30,
    repeats: int = 16,
    target_improvement: float = 0.10,
) -> dict:
    if final_step < 0 or eval_every <= 0:
        raise ValueError("Final step must be nonnegative and evaluation interval must be positive")
    if not math.isfinite(target_improvement) or not 0 < target_improvement <= 1:
        raise ValueError("Target improvement must be finite and in (0, 1]")
    baseline, baseline_release = read_scores(results / "eval-0000.jsonl")
    expected = {key[0] for key in baseline}
    if len(expected) != questions or any(
        {seed for question, seed in baseline if question == question_id} != set(range(repeats))
        for question_id in expected
    ):
        raise ValueError("Baseline does not match the declared question count and seeds 0 through repeats-1")
    curve = []
    final = None
    for path in sorted(results.glob("eval-*.jsonl")):
        step = int(path.stem.removeprefix("eval-"))
        scores, release = read_scores(path)
        if baseline.keys() != scores.keys():
            raise ValueError(f"Evaluation question/seed sets differ at step {step}")
        curve.append({"step": step, "release_id": release, "accuracy": sum(scores.values()) / len(scores)})
        if step == final_step:
            final = scores
            if final_step and release == baseline_release:
                raise ValueError("Final checkpoint still identifies the baseline release")
    if final is None:
        raise ValueError(f"No evaluation for the predeclared final step {final_step}; acceptance is pending")
    expected_steps = {0, final_step, *range(eval_every, final_step, eval_every)}
    observed_steps = [point["step"] for point in curve]
    missing = sorted(expected_steps - set(observed_steps))
    unexpected = sorted(set(observed_steps) - expected_steps)
    if len(observed_steps) != len(set(observed_steps)):
        raise ValueError("Duplicate evaluation steps; acceptance is pending")
    if missing or unexpected:
        raise ValueError(
            f"Evaluation schedule is incomplete or mismatched: missing={missing}, unexpected={unexpected}; "
            "acceptance is pending"
        )
    improvement = (sum(final.values()) - sum(baseline.values())) / len(baseline)
    return {
        "final_step": final_step,
        "eval_every": eval_every,
        "questions": questions,
        "samples_per_question": repeats,
        "target_absolute_improvement": target_improvement,
        "absolute_improvement": improvement,
        "target_met": improvement >= target_improvement,
        "observed_positive_improvement": improvement > 0,
        "paired_question_bootstrap_95_interval": paired_interval(baseline, final),
        "uncertainty_note": "Question-cluster bootstrap, 10,000 resamples, seed 0; the point-estimate target is separate from statistical significance.",
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--final-step", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--target-improvement", type=float, default=0.10)
    parser.add_argument("--training-mode", choices=["full-parameter", "LoRA"], default="full-parameter")
    args = parser.parse_args()
    report = analyze(
        args.results,
        final_step=args.final_step,
        eval_every=args.eval_every,
        target_improvement=args.target_improvement,
    )
    (args.results / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    curve = report["curve"]
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot([point["step"] for point in curve], [100 * point["accuracy"] for point in curve], marker="o")
    axes.axhline(
        100 * (curve[0]["accuracy"] + args.target_improvement),
        linestyle="--",
        color="gray",
        label=f"SFT baseline + {100 * args.target_improvement:g} pp",
    )
    axes.set(
        xlabel="OPD optimizer updates",
        ylabel="AIME'24 mean accuracy (%)",
        title=f"Qwen3.5-9B {args.training_mode} OPD",
    )
    axes.legend()
    axes.grid(alpha=0.2)
    figure.savefig(args.results / "learning-curve.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
