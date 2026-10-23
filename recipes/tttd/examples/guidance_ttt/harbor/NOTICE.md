# Benchmark sources

- Polyomino Packing: FrontierCS problem 0 at `6d597dfb60be9e592881aef051b94e30d197c436`.
  The existing Reef example supplies its instruction and initial program.
- AHC058: AtCoder's Apple Production Planning task. The public inputs, tester,
  and initial solution come from TTT-Discover at `6c40e82dab9d5de7416ac873ad5cd3106084aaed` (MIT).
  ALE-Bench is pinned in `../judges/requirements-ale-bench.txt`.
- TriMul: TTT-Discover at `6c40e82dab9d5de7416ac873ad5cd3106084aaed` (MIT).
  The instruction describes the outgoing AlphaFold3 operation. Preparation
  downloads its initial solution and official evaluator with the upstream license.
- Lasso: SimpleTES at `47d3413da1d85dc24341219d47452d2601e56a57` (AGPL-3.0).
  Preparation downloads the original bootstrap and evaluator separately.
  The task contract documents their input/output format and correctness conditions.
- FrontierCS 2 (`join_tree_rewriting`, `approximate_mips_query_stopping`,
  `lejepa_world_model`, `polar_express`): FrontierCS 2 at
  `e989e8c6826585a336fb999db1b87fa327073c2d` (CC BY 4.0). Each `instruction.md`
  carries the task's research statement unchanged after a candidate-format
  section. Preparation checks out the task repositories, and the judge runs the
  task's published verifier image.
