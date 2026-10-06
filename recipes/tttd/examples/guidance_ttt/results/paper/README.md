# Paper Results and Search Trajectories

This directory contains the reported results and search trajectories for the four Guidance-TTT tasks.
The [solutions directory](../../solutions/README.md) provides complete programs, a result overview, and evaluation commands.

## Available Data

| File | Contents |
|---|---|
| [results.json](results.json) | Main results, baseline comparisons, models, metrics, evaluation suites, and search budgets |
| [polyomino_score_trajectory.csv](polyomino_score_trajectory.csv) | Polyomino scores across training updates |
| [trimul_14b_frontier.csv](trimul_14b_frontier.csv) | TriMul best-so-far latency during search |
| [search_histories.csv](search_histories.csv) | Lasso and AHC best-so-far search scores |

The [four-task trajectory figure](../../../../../../docs/assets/guidance-ttt/best-solution-trajectories.png)
shows score progression and selected algorithm changes.

## Read the Metrics

Search trajectories record evaluation results during optimization. Final results use the paper's fixed-program evaluations or official submissions.

- **Polyomino:** higher packing scores are better.
- **Lasso:** the search peak is about 0.21259. The final five-evaluation mean is 0.1739.
- **AHC058:** the search figure uses the total over 150 public cases. The final result is the official AtCoder score.
- **TriMul:** lower latency is better. Search-time latency and final repeated measurements describe different evaluation runs.

Baseline entries retain their source models and budgets, as recorded in `results.json`.
The [earlier Reef runs](../README.md) use different configurations and remain separate from these paper results.

## Source Records

The main-result data follows [paper revision 709c405](https://github.com/Chonghe-Jiang/Guidance-ttt-paper/tree/709c405).
`results.json` records the source checksums and displayed precision.
CSV files use LF line endings. Source checksums refer to the original files before newline normalization.
