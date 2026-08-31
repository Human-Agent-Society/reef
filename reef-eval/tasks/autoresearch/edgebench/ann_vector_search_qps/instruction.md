## Role

You are an expert in approximate nearest neighbor (ANN) search. Your job is to maximize **QPS** (queries per second) on the SIFT-1M dataset (1M × 128-d vectors, 10K queries) under the ann-benchmarks framework, while keeping **Recall@10 ≥ 0.95**.

---

## Repository Layout

- `run.py` — evaluation driver (already present)
- `ann_benchmarks/algorithms/` — algorithm plugins; each algorithm subclasses `BaseANN` and registers in its own `config.yml`
- `ann_benchmarks/algorithms/custom/` — **your** slot; contains a baseline `module.py` (brute-force numpy) and a `config.yml` naming the algorithm `custom`
- `data/sift-128-euclidean.hdf5` — dataset (pre-downloaded, contains `train` and `test` arrays)

---

## What To Do

1. Read `README.md`, `ann_benchmarks/algorithms/README.md`, and look at existing algorithms (e.g. `ann_benchmarks/algorithms/faiss/`) for reference implementations.
2. Improve `ann_benchmarks/algorithms/custom/module.py` — implement a faster `fit()` + `query()` using FAISS (`faiss-cpu` is pre-installed) or a custom implementation. Keep the class name `Custom` and the module path `ann_benchmarks.algorithms.custom`.
3. You may tune hyperparameters via `ann_benchmarks/algorithms/custom/config.yml` (IVF nlist, HNSW M/efConstruction, PQ subquantizers, etc.).
4. For local runs after each change:
   ```bash
   python run.py --local --algorithm custom --dataset sift-128-euclidean -k 10 --runs 1 --run-disabled
   ```
   The benchmark prints QPS and recall metrics to the console. Use these to track your progress.

---

## Optimization Directions (suggestions)

- IVF bucketing (nlist, nprobe)
- HNSW (M, ef, efConstruction)
- Product Quantization (subquantizers, bits)
- Two-level routing (IVF-HNSW, IVF-PQ)
- Early termination on candidate scan
- Multi-threaded search (omp_set_num_threads)

GLM-5.1 reference: 3108 QPS → 21472 QPS (6.9×) over 655 iterations.

---

## Scoring Integrity (CRITICAL)

The evaluation system independently measures query latency and verifies search results against the ground truth. Your score is determined by the actual search performance of your algorithm.

You must NOT:
- Read or access ground truth data (the `neighbors` dataset) during query execution
- Override timing attributes or manipulate benchmark measurements (e.g., via `get_additional()`)
- Monkey-patch the `time` module or any benchmark framework internals
- Attempt to fabricate or inflate scores through any means other than legitimate algorithmic improvements

Any attempt to manipulate scores will result in score=0 and the submission marked as hack.

---

## Rules

- Keep the algorithm name `custom` and leave its config discoverable via `python run.py --list-algorithms`
- Do NOT modify `run.py` or any file outside `ann_benchmarks/algorithms/custom/`
- Recall@10 < 0.95 on a config makes that config invalid (score = 0)
- Final score is the **highest QPS** among valid (recall-passing) configs
