# gepa

Reproduction of [GEPA](https://arxiv.org/abs/2507.19457) as a Reef harness-evolution recipe. GEPA keeps an archive of candidate prompts, Pareto-samples a parent, and reflects on its training-minibatch traces to rewrite one component. A child survives only if it beats its parent on that minibatch, then receives a full validation pass; Reef serves the candidate with the best mean validation score. The package holds the method; the loop that validates it lives in [examples/aime](examples/aime/README.md).

- Paper: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- Pins: upstream GEPA v0.1.2 (`92dadfffbe98c8ecf508179a1cab09c1bb85cd32`), Pi `0.84.2`, task model `gpt-4.1-mini-2025-04-14`, and reflection model `gpt-5-2025-08-07`; the example pins and hash-checks the 45/45/150 AIME train/validation/test split and uses a 150-metric-call search budget. The method has no upstream GEPA runtime dependency.
- Claim scope: the [AIME-2025 validation contract](examples/aime/README.md#the-validation-contract) pairs two seeds with upstream GEPA under the same models, split, budget, scorer, and worker count. Mean held-out improvement is +16.00 percentage points for this method and +12.67 for upstream; the selected-score difference changes sign between seeds, so these results validate the method but do not establish superiority. Deterministic tests cover the scorer and driver, with an upstream fidelity comparison when GEPA is installed.

## Layout

```text
gepa/
  archive.py      candidates, Pareto fronts, seeded sampling, and search budget
  backend.py      committed algorithm state and the post-commit archive mirror
  components.py   harness components as prompt texts and mutations
  method.py       reflection proposals, minibatch acceptance, and selection
  reflection.py   attributed upstream reflection prompt and record formatting
  recipe.py       GEPARecipe: configuration and evolution backend binding
  examples/aime/  the runnable validation loop: harness, gepa.yaml, and results
```

## Where the rest is documented

[The gepa recipe page](../../docs/user-guide/recipes/gepa.rst) covers the algorithm and configuration, [Evolve your harness](../../docs/user-guide/evolve-your-harness.rst) describes the evolution engine, and the [example README](examples/aime/README.md) records setup, implementation details, distance from the upstream protocol, and the completed validation.
