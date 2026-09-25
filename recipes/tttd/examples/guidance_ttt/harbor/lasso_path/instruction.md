# Optimize a Gaussian Lasso path solver

Write a complete Python module containing a raw string named `CPP_CODE`.
The string must define a standalone C++17 solver. An optional `COMPILE_FLAGS`
list supplies additional compiler flags. Keep `# EVOLVE-BLOCK-START` and
`# EVOLVE-BLOCK-END` around these definitions.

For each decreasing regularization value lambda, minimize
`||y - Xw||^2 / (2*n) + lambda * ||w||_1`.
Return one coefficient vector for every lambda. Use float64 throughout.

The executable reads this binary stream from standard input, in order:

1. Three int32 values: `n`, `p`, and `n_lambda`.
2. `n*p` float64 values for the row-major matrix `X`.
3. `n` float64 values for `y`.
4. `n_lambda` float64 values for the decreasing regularization path.

Write exactly `p*n_lambda` float64 coefficients to standard output.
Use column-major order: each consecutive block of `p` values corresponds to
one lambda. Do not print diagnostics to standard output.

The trusted evaluator compiles the program and runs 17 synthetic cases.
Every solution objective must be at most the scikit-learn reference objective
plus `1e-6`. A correctness failure makes the candidate invalid.
The score is the reciprocal of the geometric-mean solve time in milliseconds.
Higher scores are better.

The initial program uses warm starts, screening, coordinate descent, and KKT
checks. Improve its algorithm or implementation while preserving the full
input/output contract. Do not modify the evaluator, read external files,
use network access, cache answers across inputs, or substitute lower precision.
