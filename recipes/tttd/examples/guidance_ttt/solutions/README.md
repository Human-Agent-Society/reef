# Discovered Solutions

This directory contains the complete solutions discovered by Guidance-TTT for the four tasks in our paper.
Each solution can be evaluated independently without training or model API calls.

| Task | Guidance Model | Solution | Reported Result |
|---|---|---|---|
| Polyomino Packing | Qwen3-8B | [C++17](polyomino_qwen3_8b.cpp) | 91.89 score ↑ |
| Lasso Path | Qwen3-8B | [Python wrapper + C++17](lasso_qwen3_8b.py) | 0.1739 ms⁻¹ ↑ |
| AHC058 | Qwen3-14B | [C++20](ahc058_qwen3_14b.cpp) | 850,082,731 AtCoder score ↑ |
| TriMul | Qwen3-14B | [Python/Triton](trimul_qwen3_14b.py) | 1,129 µs ↓ |

All four searches used GLM-5.2 as the frozen executor.
The arrows indicate whether higher or lower values are better.
The [results directory](../results/paper/README.md) contains comparison data and search trajectories.

## Evaluate a Solution

Use Python 3.12 or newer and the [example dependencies](../README.md#run).
Start the selected judge with the [evaluator setup instructions](../judges/README.md).
From `recipes/tttd/examples/guidance_ttt/`, run the matching command:

```bash
python evaluate_solution.py polyomino_packing --judge-url http://127.0.0.1:8081
python evaluate_solution.py lasso_path --judge-url http://127.0.0.1:8082 --repeats 5
python evaluate_solution.py ahc058 --judge-url http://127.0.0.1:8082
python evaluate_solution.py trimul --judge-url http://127.0.0.1:8082 --repeats 3
```

The examples reuse port 8082 for different judges. Each command requires the matching task judge on that port.
TriMul requires an H100 evaluator. Cloud evaluation can incur charges.

Each command runs the requested evaluations sequentially.
It prints one JSON record per evaluation:

- `result.valid`: whether the solution passed the required correctness checks.
- `result.artifacts.score_unbounded`: the raw task metric described below.
- `result.score`: the training reward, which differs from the raw metric for AHC058 and TriMul.

Exit codes are `0` for all-valid results, `1` for an invalid solution, and `2` for a judge infrastructure error.

## Evaluation Details

The table reports the paper measurements.
Evaluation results can vary across runs due to task-specific randomness and timing noise.
Hardware, software, and thread settings also affect the measurements.

- **Polyomino Packing:** packing score across the 70-case FrontierCS suite.
- **Lasso Path:** inverse geometric-mean solve time across 17 synthetic cases, subject to the evaluator's correctness tolerance.
  The reported result averages five evaluations on an AMD EPYC 9745 with one thread.
  The default judge uses a four-core allocation, so matching the paper also requires the single-thread configuration.
- **AHC058:** the table reports the official AtCoder score.
  The local evaluator returns the mean raw score over 150 public cases, a different metric.
- **TriMul:** geometric-mean latency across seven timing cases after all correctness checks pass.
  The reported result averages three evaluations on an H100 80GB PCIe with CUDA 12.8, PyTorch 2.7.1, and Triton 3.3.1.

The [manifest](manifest.json) lists program filenames, reported results, and evaluation settings in machine-readable form.

## License

The programs use the repository license. External evaluators retain their own licenses, as listed in the [benchmark notices](../harbor/NOTICE.md).
