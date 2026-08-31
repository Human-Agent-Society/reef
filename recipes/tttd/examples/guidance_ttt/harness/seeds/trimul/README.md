# TriMul scratch seed

`glm52_scratch_bootstrap_library.json` is the fixed default root for Reef's
summary-only TriMul task. GLM-5.2 generated it once from only the public problem
statement: no parent candidate, prior solution, search history, or published
TTT-Discover kernel was present in the prompt.

The pinned official H100 evaluator recorded 18/18 passing correctness cases,
7/7 passing benchmark cases, geometric-mean runtime
`10177.396849081848 us`, and reward `1500 / runtime_us =
0.14738542893071152`. The solution SHA-256 is
`49485cfe6f1e0f5d2ea40df2bead246b59ab8774cff3a79ab03cba4cf08995a9`.

The JSON preserves the generation prompt/response, complete candidate,
canonical summary, API metadata, and official verifier artifacts. The runner
copies this pristine archive into each run and validates its file digest on
resume.
