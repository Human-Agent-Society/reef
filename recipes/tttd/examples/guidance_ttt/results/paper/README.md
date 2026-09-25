# Four paper results

`results.json` transcribes the main table in `sections/case_study.tex` from
[paper commit 709c405](https://github.com/Chonghe-Jiang/Guidance-ttt-paper/tree/709c405).
It records models, metric units, evaluation suites, budgets, baselines, and source SHA-256 checksums.
The values retain the paper's displayed precision.

These results are historical measurements. They do not establish that this
new port reproduces those values. Baseline models and budgets differ.

| Task | Guidance / frozen executor | Final reported result |
|---|---|---|
| Polyomino Packing | Qwen3-8B / GLM-5.2 | 91.89, 70-case score |
| Lasso Path | Qwen3-8B / GLM-5.2 | 0.1739, inverse geometric-mean solve time |
| AHC058 | Qwen3-14B / GLM-5.2 | 850,082,731, AtCoder score |
| TriMul | Qwen3-14B / GLM-5.2 | 1,129 microseconds, geometric-mean H100 latency |

The [guide's four-panel figure](../../../../../../docs/assets/guidance-ttt/best-solution-trajectories.png)
is the paper's original `figures/best_solution_trajectories.png`, copied without changes.
It annotates algorithm changes and distinguishes search scores from final reported results.
`polyomino_score_trajectory.csv` and `trimul_14b_frontier.csv` preserve the numerical
histories available beside that paper figure. Lasso and AHC histories are in `search_histories.csv`.

Lasso's search peak is about 0.21259, not its final 0.1739 result.
AHC058's search figure uses the total over 150 public cases, not the final AtCoder score.
TriMul's search curve and fixed-kernel measurements are separate observations.
The earlier files under the parent `results/` directory describe different Reef runs.

CSV files use LF line endings. Source checksums refer to the original paper files, before newline normalization.
