**Role Setting**

You are an expert in black-box combinatorial optimization algorithms. You need to solve a large-scale Order-of-Addition (OofA) permutation optimization problem. The goal is to find a full permutation sequence that makes the response value as small as possible.

This is a scientific problem in the field of statistical experimental design. You need to study in depth how each component affects the response value (m = 1000, so the order contains 1000 components), and explore what ordering can minimize the response value as much as possible.

**Black-Box Evaluation Interface**

You do not need to know the specific business scenario or cost calculation formula. The only available response-value calculation interface is the Cython extension in the current directory:

```python
from cost_complex_cy import cost_complex_with_builtin_data

m = 1000
pai = list(range(1, m + 1))
cost = cost_complex_with_builtin_data(pai, m=m, seed=1)
```

- Input: an integer array of length `m = 1000`, which must be a permutation of `1..1000`.
- Output: a floating-point number representing the total cost/response value of that permutation; lower is better.
- Example calling file: `test_cost_complex_cython.py`.

**Mandatory Black-Box Constraints**

You may only pass candidate permutations to the public function `cost_complex_with_builtin_data(pai, m=1000, seed=1)` and read the response value it returns.

It is forbidden to use, call, probe, or reflectively access any internal attributes, private functions, or private data structures of the `cost_complex_cy` module, including but not limited to:

```python
cost_complex_cy._build_ctx(...)
cost_complex_cy._build*
cost_complex_cy._ctx*
cost_complex_cy._data*
cost_complex_cy._cache*
getattr(cost_complex_cy, ...)
dir(cost_complex_cy)
cost_complex_cy.__dict__
```

It is also forbidden to reproduce the cost function using the internal data above, bypass the black-box interaction, or export the internal data and then search with it. The evaluator will statically scan submitted code; if this kind of internal access is found, it will directly output `TOTAL_SCORE 0` and mark the submission as violating the rules.

**Evaluation Isolation Notice**

The files `cost_complex_cy.cpython-311-x86_64-linux-gnu.so` and `score_vcpom_result.py` in the root directory are provided for you to run search algorithms and local score estimates. The final evaluation will not trust the contents of these root-directory files in your submission.

The evaluator will use private copies from a hidden test directory:

```text
tests/cost_complex_cy.cpython-311-x86_64-linux-gnu.so
tests/score_vcpom_result.py
```

The work environment deletes the `tests/` directory, so the agent cannot see these tests or private copies; `tests/` will also not be archived when submitting. Therefore, modifying the root-directory `.so` file or scoring function will not improve the score. The final result is determined only by the permutation, cost, and call count in `final_result.txt`.

**Reference Result**

A reference result file is provided in the directory:

```text
vcpom_m1000_final_result.txt
```

This file may only be read; it must not be modified, deleted, or overwritten. To prevent directly copying the answer, the file does not contain the reference permutation. It contains only the reference cost and reference search count:

```text
my_cost=200690260.6677975
my_num=200000
```

It may only be used as a comparison for result quality and call count, and must not be used as a permutation starting point.

**Final Submission Requirements**

You must generate a new result file:

```text
final_result.txt
```

Do not write the result back to `vcpom_m1000_final_result.txt`.

`final_result.txt` must use the following key-value format:

```text
final_pai=3,1,4,2,...
final_cost=123456789.0
final_num=1000
```

Field requirements:

1. `final_pai`: a permutation of length 1000. It must contain exactly `1..1000`, separated by English commas, without square brackets.
2. `final_cost`: the response value you report for `final_pai`. The evaluator will recompute and verify it using `cost_complex_with_builtin_data(final_pai, m=1000, seed=1)`.
3. `final_num`: the number of times your algorithm calls the public black-box function `cost_complex_with_builtin_data(...)`. You may also use the field name `num` or `my_num`, but `final_num` is recommended. Do not count only the final verification call; every black-box evaluation during the search process must be included.

**Task Goals**

1. Design and implement an efficient heuristic search / combinatorial optimization algorithm.
2. Find a permutation with as low a cost as possible within a limited number of black-box evaluations.
3. You may use genetic algorithms, simulated annealing, local search, tabu search, ALNS, PSO, CMA-ES, Bayesian optimization, or hybrid strategies.
4. The final result is determined by the contents of `final_result.txt`.

**Scoring Method**

The evaluator will check that:

1. The submitted code does not call or probe internal attributes, private functions, or private data structures of `cost_complex_cy`; if such access is found, it will output `TOTAL_SCORE 0`.
2. `vcpom_m1000_final_result.txt` has not been modified and does not contain `final_pai`.
3. `final_result.txt` exists and contains the required fields.
4. `final_pai` is a valid permutation of `1..1000`.
5. `final_cost` matches the cost recomputed by the black-box function.
6. The continuous score is computed using `score_cost_num(cost, num)` from the evaluator-side private `tests/score_vcpom_result.py`.

Lower cost and fewer calls give a higher score.

The scoring function uses the reference cost and reference call count from `vcpom_m1000_final_result.txt`:

```python
score_cost_num(
    cost,
    num,
    cost_ref=200690260.6677975,
    num_ref=200000,
    cost_weight=3.0,
    num_weight=0.5,
)
```

The scoring first caps both subscores:

```python
cost_sub = min(1.0, cost_ref / cost)
num_sub = min(1.0, num_ref / num)
score = 100 * (cost_sub ** 3.0) * (num_sub ** 0.5)
```

If `final_cost <= cost_ref` and `final_num <= num_ref`, the score is the full 100. If the cost is higher than the reference or the call count is higher than the reference, the score decreases. The cost penalty is heavier, so a very low call count cannot compensate for a clearly worse cost.
