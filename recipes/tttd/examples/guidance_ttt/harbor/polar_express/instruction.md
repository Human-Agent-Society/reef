<!-- From FrontierCS 2 (https://github.com/FrontierCS/FrontierCS-2, CC BY 4.0),
task `polar_express`; the statement after the first section is unchanged. -->

# Candidate format

The candidate is the complete new text of `polar_express.py`. The judge writes it over
that file in the task repository, makes the task's patch, and scores it with the
task's own FrontierCS 2 verifier on the development (train) suite. Every other
file of the repository stays unchanged; this run does not submit helper files.
The verifier message returns the train score and the development metrics that a
FrontierCS 2 agent receives, nothing else.

# Polar-Factor Approximation

## Optimization priority

Your primary objective is **PASS on every required workload and every required
metric**, with all correctness and validity requirements satisfied. The continuous
score is a secondary optimization objective; a higher score cannot compensate for
any failed PASS condition. Do not replace a solution known to satisfy all tested
PASS conditions with one known to fail them merely to increase its score.
Exploratory trials may regress; if no solution passes yet, work on the unmet
conditions and retain a valid submission as a fallback.

Successful verifier execution or a positive score does not mean scientific PASS.
Use the reported per-metric pass/fail progress to assess the tested workloads.
Train feedback covers only its evaluated workloads; passing train does not prove
final PASS. Final independently evaluates the submitted artifact on its specified
suite. The scoring formula, validity rules, and PASS comparisons below are unchanged.

## Problem
For a real matrix with reduced singular-value decomposition \(G=U\Sigma V^T\), the polar factor is \(UV^T\), the closest matrix of the same shape with orthonormal columns when \(G\) is tall or orthonormal rows when it is wide. Polar factors are used in numerical linear algebra and as matrix-orthogonalisation operations in optimisers for neural-network training. The substrate is an extract of the open-source PolarExpress numerical artifact: it retains the project's Python module and a polynomial-visualisation notebook. The program receives interval and iteration parameters for odd-polynomial schedules, plus float32 PyTorch batches of synthetic square, tall, and wide matrices. A suite contains twelve matrix members: six original members with log-spaced singular values ranging from \(10\times18\) to \(18\times10\) (1,008 float32 matrix entries, schedules of three to five steps), and six paper-regime members of shape \(256\times256\), \(128\times512\) and \(512\times128\) with five or six steps and hidden singular spectra. The three Python interfaces return finite coefficient sequences and finite tensor approximations; each tensor result must have the input shape, contain one polar-factor approximation per batch member, and leave the input unchanged. Their return values are captured by the numerical worker and passed to the judge-side scorer; the program writes no scored output file. The secondary continuous objective is to maximise the 0--100 score obtained by equally weighting scalar composition accuracy and matrix polar-factor accuracy. The bodies of `optimal_quintic`, `optimal_composition`, and `PolarExpress` have been removed from `polar_express.py` and must be supplied.

## Design space
The editable surface consists of the three removed numerical interfaces and Python helpers placed under `frontier_solution/`. A submission may replace the stubs, change other content in `polar_express.py`, and organise supporting Python code inside that helper directory. The contract consists of the values returned by `optimal_quintic` and `optimal_composition`, the tensor returned by `PolarExpress`, successful execution of the retained notebook, and the validity rules below. The evaluator calls only these interfaces; there is no solver-owned command-line program, service, output file, training loop, notebook edit, dataset lookup, or GPU component. Only the dependencies listed under Resources are provisioned.

## Workload
Each development or graded suite contains the following twelve workload records, in this order. The first six are the original records; the last six form the paper-regime stratum:

| `id` | `shape` | `l` | `steps` | `safety` | `cushion` |
| --- | --- | ---: | ---: | ---: | ---: |
| `square_0` | `[12, 12]` | 0.010 | 5 | 0.010 | 0.020 |
| `square_1` | `[12, 12]` | 0.020 | 5 | 0.008 | 0.015 |
| `tall_0` | `[18, 10]` | 0.030 | 4 | 0.006 | 0.012 |
| `tall_1` | `[18, 10]` | 0.050 | 4 | 0.004 | 0.010 |
| `wide_0` | `[10, 18]` | 0.080 | 3 | 0.003 | 0.008 |
| `wide_1` | `[10, 18]` | 0.120 | 3 | 0.002 | 0.006 |
| `paper_square_0` | `[256, 256]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_square_1` | `[256, 256]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_wide_0` | `[128, 512]` | 0.001 | 6 | 0.010 | 0.02407327424182761 |
| `paper_wide_1` | `[128, 512]` | 0.001 | 6 | 0.010 | 0.02407327424182761 |
| `paper_tall_0` | `[512, 128]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_tall_1` | `[512, 128]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |

Here is one actual record in its evaluator layout:

```json
{"id":"tall_0","shape":[18,10],"l":0.030,"steps":4,"safety":0.006,"cushion":0.012}
```

`id` labels the separately scored workload; `shape` is `[rows, columns]`; `l` is the inclusive lower endpoint of the scalar interval `[l, 1.0]`; `steps` is both the requested coefficient-sequence length and matrix iteration count; and `safety` and `cushion` are passed as `safety_factor_eps` and `cushion`. These values, along with the case count and ordering, are challenge constants (not from materials).

For an original record with shape `[m,n]`, let `k=min(m,n)`. Following the artifact's numerical experiment, the evaluator draws independent standard-normal arrays, obtains reduced orthonormal factors \(Q_L\in\mathbb{R}^{m\times k}\) and \(Q_R\in\mathbb{R}^{n\times k}\), sets `singular = np.geomspace(l, 1.0, k)`, and forms \(G=(Q_L\,\mathrm{diag}(\text{singular}))Q_R^T\). It converts `G` to float32. Adjacent cases of the same shape are stacked, so the actual tensor record received by `PolarExpress` is indexed as `batch[member, row, column]` and has shape `[2,m,n]`; each member remains a separate scored workload. The six original records therefore make three calls carrying six matrix members. A float64 reduced SVD of each generated `G` defines the exact target \(UV^T\).

The six paper-regime records follow the source paper's convergence experiments: larger matrices, the five or six iterations used with Muon, and spectra that the method must handle without knowing them. Their `l`, `safety` and `cushion` values (the paper's offline defaults) parameterise only the scalar check; the matrix spectrum is not tied to `l`. Each paper-regime member draws its singular values independently, from a generator stream separate from the original records, out of three hidden families: log-spaced spectra with \(\sigma_{\min}/\sigma_{\max}\) between about \(10^{-4.5}\) and \(10^{-2.5}\), power-law decay, and low effective rank with a small tail. The spectrum is then multiplied by an arbitrary global scale. Shape and step count therefore do not identify the spectrum, and the two members of a pair need not share a family. They are stacked in pairs exactly like the original records, so a complete suite makes six `PolarExpress` calls carrying twelve members, and every member is a separate graded workload.

For every record, the scalar check uses the inclusive 1,000-element float64 array `np.linspace(l, 1.0, 1000)`. The evaluator calls `optimal_quintic(l, 1.0)` and `optimal_composition(l, steps, degree=5, safety_factor_eps=safety, cushion=cushion)` before calling the matrix interface. Across the twelve records, the direct composition output contains 56 steps and 168 coefficients. The retained notebook is also executed in full. Its module import evaluates `optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02)`, and its plotting cell calls `optimal_composition(0.001, 12, degree=5, safety_factor_eps=0.01, cushion=0.02)` once through `plot_composition_arrows(l1=0.001, n_steps=12, degree=5)`.

The cases are representative generated numerical inputs, not values to recognise. Development and grading use disjoint random seeds for the orthonormal factors while retaining the same twelve-record stratification, dimensions, intervals, original spectra and paper-regime spectrum families, parameters, ordering, and one execution of each case per evaluator call. The paper-regime spectra are drawn afresh from those families for each seed.

## Interface
There is no solver-facing command-line input or textual result stream. The evaluator imports `polar_express` from the repository root and calls these Python interfaces; their return values are the complete program output consumed by the score.

Before evaluation, the build adapter runs `python -m py_compile polar_express.py` followed by `python -c 'import polar_express'` from the repository root.

`optimal_quintic(l, u)` receives numeric interval endpoints satisfying `0 <= l <= u` in all required calls. It must return exactly three finite real values `(a, b, c)`, in increasing power order, representing

\[
p(x)=a x+b x^3+c x^5.
\]

`optimal_composition(l, num_iters, degree=5, safety_factor_eps=0, cushion=0)` receives the initial lower endpoint, requested sequence length, polynomial degree, and two numeric safeguard parameters. In the directly scored degree-5 calls, it must return exactly `num_iters` finite three-coefficient sequences in execution order. The retained notebook calls it with degree 5 and iterates over the returned object, so that call must supply coefficient sequences compatible with the same odd-power evaluator and complete without error. No retained notebook call exercises degree 3. A degree-3 sequence conventionally has `(a,b)` for \(a x+b x^3\); a degree-5 sequence has `(a,b,c)` for \(a x+b x^3+c x^5\). For scoring, a grid value is transformed once by each returned degree-5 sequence, from first to last.

`PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor` receives a float32 tensor `G` of shape `[2,m,n]` and the positive step count listed in the workload table. It must return a `torch.Tensor` with exactly the same shape, all values finite, and one polar-factor approximation per batch member. It must not mutate any element of `G`.

The following concrete calls show the layouts. The coefficient values shown are valid finite records that make the formats explicit; their numerical errors, rather than mere validity, determine the score.

```python
>>> optimal_quintic(0.5, 1.0)
(1.0, 0.0, 0.0)               # a=1, b=0, c=0: p(x)=x

>>> optimal_composition(0.5, 2, degree=5,
...                     safety_factor_eps=0.0, cushion=0.0)
[(1.0, 0.0, 0.0),
 (1.0, 0.0, 0.0)]             # two steps, in application order

>>> G = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
>>> PolarExpress(G, steps=2)
tensor([[[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]]])           # shape [batch=2, rows=2, columns=2]
```

The judge executes `plotsforpolar.ipynb` from the repository directory using `NotebookClient(timeout=60, kernel_name="python3", allow_errors=False)`. It then imports the module in the numerical runner and performs the calls above with the twelve exact records. Execution sets `MPLBACKEND=Agg`, `TORCH_COMPILE_DISABLE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONHASHSEED=0`; PyTorch is restricted to one thread in the direct numerical checks.

## Validity
Patch validation is applied before Python execution. The artifact must be a non-empty textual unified diff of at most 1,048,576 bytes, change no more than 20 files, contain no binary patch content, and touch only `polar_express.py` or paths below `frontier_solution/`. Source files must apply cleanly to the pristine tree. `polar_express.py` and every Python file below `frontier_solution/` must parse successfully.

Solver-owned Python has a deliberately narrow capability set. Imports are limited to `math`, `itertools`, `numpy`, `numpy.linalg`, `numpy.polynomial`, `numpy.polynomial.polynomial`, `torch`, and `frontier_solution` or its helper modules; other submodules, relative imports, private imported names, and wildcard imports are rejected. Source may not use dynamic lookup or execution (`getattr`, `setattr`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, or `__import__`), frame or class introspection, dunder attributes, attribute assignment, tensor export (`numpy`, `tolist`, `item`, storage or array/DLPack protocols), matrix decomposition names, or named NumPy matrix-product/decomposition entry points (including `matmul`, `dot`, `einsum`, and `lstsq`). Public access to `sys`, `os`, `builtins`, `importlib`, `subprocess`, or `pathlib` through a numerical module is also forbidden. These checks apply to attribute lookup and alias creation as well as direct calls. The supplied `torch.compile` decorator and aliases of that PyTorch API remain allowed; the built-in dynamic `compile` is forbidden. They allow NumPy's scalar/coefficient-array facilities, including array `@` and the linear solve needed by the coefficient generator, for arrays derived from scalar interval/coefficient parameters. This allowance does not permit named NumPy matrix-product/decomposition APIs or moving the scored tensor into that backend. Tensor export remains forbidden; the checker does not infer the mathematical provenance of every arbitrary array.

Candidate import and execution occur in a worker process that receives only the twelve input matrices and public workload records. It contains no SVD oracle, reference row, or private comparison/scoring state. Tensor-export methods are disabled before candidate import, so an import-time alias cannot retain them. During each guarded `PolarExpress` call, PyTorch permits only dtype conversion, view/transpose/expand operations, vector norm, elementwise add/subtract/multiply/divide, and dense `bmm`, `mm`, `matmul`, or `addmm`; every other dispatched operation is rejected. The call may dispatch at most `3 * steps` of those dense matrix products across the whole two-member batch call. Genuine `torch.matmul`, direct imported aliases, helper re-exports, `Tensor.matmul`, and bound tensor `.matmul` are supported. Attribute lookup and imports of `matmul` are checked against actual PyTorch object identities, including after variable rebinding; NumPy and arbitrary lookalike receivers are rejected. The temporary execution copy of the submitted source is read-only during execution, child processes are forbidden, and names beginning `_frontier_` are reserved for the verifier. After the worker exits, its returned coefficients and tensor bytes are validated and scored against private float64 oracles in the judge process.

The notebook must import and execute both retained code cells without an error. `optimal_quintic` must return three finite coefficients. Each directly scored degree-5 composition must have exactly the requested number of steps, and every step must contain exactly three finite coefficients; applying the entire sequence to the scalar grid must remain finite. Every matrix result must be a tensor with the input batch's exact shape, must be finite, and must leave the input bit-for-bit unchanged.

An invalid patch, parse or import failure, forbidden operation, operation-budget excess, notebook error, wrong return length, wrong coefficient count, wrong tensor type or shape, input mutation, non-finite intermediate or result, crash, or timeout scores zero before accuracy is considered. The overall evaluator timeout is 1800 seconds, each candidate numerical subprocess is limited to 100 seconds, and each notebook cell is limited to 60 seconds. The build timeout is 600 seconds.

## Scoring
The secondary continuous objective is to maximise the returned score on a scale from 0 to 100. For workload \(i\), write its returned coefficient sequence as \((c_{t,0},c_{t,1},c_{t,2})\) for \(t=1,\ldots,T_i\), define \(p_t(x)=c_{t,0}x+c_{t,1}x^3+c_{t,2}x^5\), and let \(p_{T_i}\circ\cdots\circ p_1\) denote application in returned order. On the workload's grid \(D_i=\texttt{np.linspace}(l_i,1.0,1000)\), the scalar error is

\[
e_{s,i}=\max_{x\in D_i}\left|\left(p_{T_i}\circ\cdots\circ p_1\right)(x)-1\right|.
\]

If \(X_i\) is the corresponding member returned by `PolarExpress` and \(U_iV_i^T\) is the float64 SVD oracle, the matrix error is

\[
e_{p,i}=\frac{\lVert X_i-U_iV_i^T\rVert_F}{\lVert U_iV_i^T\rVert_F}.
\]

The continuous score uses only the six original workloads, \(i=1,\ldots,6\) in the table order; the six paper-regime members are graded only by the separate criterion below.

The evaluator computes, symbol for symbol,

\[
q_{s,i}=\min\!\left(1,\max\!\left(0,\frac{-\log_{10}(\max(e_{s,i},10^{-15}))}{12}\right)\right),
\]

\[
q_{p,i}=\min\!\left(1,\max\!\left(0,\frac{-\log_{10}(\max(e_{p,i},10^{-15}))}{8}\right)\right),
\]

\[
\mathrm{score}=100\times\frac{\frac{1}{6}\sum_{i=1}^{6}q_{s,i}+\frac{1}{6}\sum_{i=1}^{6}q_{p,i}}{2}.
\]

The error floor \(10^{-15}\), quality divisors 12 and 8, equal weighting, and clipping bounds are challenge constants (not from materials). The score is deterministic for a fixed suite, is clipped to `[0,100]`, and is returned after every accepted submission.

## Calibration
The unmodified substrate is intentionally incomplete: importing `polar_express.py` reaches a removed function and therefore the unmodified tree is not a valid submission and scores zero. A separately measured minimal validity-complete implementation scores `6.5521716654` on the full development suite. It was run three times in the declared environment and all three scores were identical, so the observed run-to-run score spread is zero. A change of `0.1` score points is treated as a genuine improvement; this interpretation is a challenge constant (not from materials).

The scale is logarithmic. Before clipping, every decimal digit of reduced scalar error contributes `1/12` to scalar quality, and every decimal digit of reduced matrix error contributes `1/8` to matrix quality. For example, scalar error \(10^{-6}\) gives scalar quality `0.5`, while matrix error \(10^{-4}\) gives matrix quality `0.5`. A submission attaining those two errors on every one of the six cases would score `50`. Errors of one or greater contribute zero for their component, while errors at or below the floor saturate that component at one. These landmarks follow directly from the scoring formula and do not alter the validity requirements.

## Feedback and grading
The judge returns a score after every submission. The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its seeds are disclosed. The development and graded suites have the same twelve workload specifications and the same distributions of generated orthonormal factors and paper-regime spectra, but use disjoint seeds and therefore share no generated matrix.

Development feedback exposes six candidate-only means: `worst_case_scalar_error` is the arithmetic mean of the six original \(e_{s,i}\) values; `relative_polar_factor_error` is the arithmetic mean of the six original \(e_{p,i}\) values; `paper_regime_relative_polar_factor_error` is the arithmetic mean of \(e_{p,i}\) over the six paper-regime members; and `spectral_polar_error`, `polar_cosine_similarity` and `top_subspace_polar_error` are the arithmetic means of \(e_{\sigma,i}\), \(c_i\) and \(e_{\top,i}\) (defined below) over all twelve workloads. Workload identifiers and other private audit fields are withheld.

A separate, stricter criterion is reported in addition to the continuous score. It compares, independently on every one of the twelve graded workloads, five quantities against the withheld method: the worst-case scalar sign-approximation error \(e_{s,i}\); the relative Frobenius error \(e_{p,i}\); the spectral-norm error \(e_{\sigma,i}=\lVert X_i-U_iV_i^T\rVert_2\); the Frobenius cosine similarity \(c_i=\langle X_i,U_iV_i^T\rangle_F/(\lVert X_i\rVert_F\lVert U_iV_i^T\rVert_F)\), with a zero output counting as 0; and the top-subspace error \(e_{\top,i}=\lVert U_1U_1^TX_iV_1V_1^T-U_1V_1^T\rVert_F/\lVert U_1V_1^T\rVert_F\), where \(U_1,V_1\) hold the oracle singular vectors whose singular values are at least \(10^{-3}\) times the largest. All four matrix quantities are computed in float64 in the judge. The criterion is satisfied only when, on every workload, each of the four candidate errors is strictly lower and the cosine similarity strictly higher than the corresponding withheld-method value. For each of the five quantities, feedback includes only aggregate counts and percentages of development workloads that pass or fail that comparison; it does not identify the workloads or disclose the withheld values. The 0--100 score remains a secondary optimisation objective, and a submission that does not satisfy the separate criterion is still scored normally.

## Resources
The runtime is a CPU-only Python 3.12 environment with 2 vCPUs, 4,096 MiB of memory, and 4,096 MiB of workspace storage. Installed numerical and notebook packages are NumPy 2.2.6, CPU PyTorch 2.8.0, Matplotlib 3.10.6, nbformat 5.10.4, nbclient 0.10.2, nbconvert 7.16.6, and ipykernel 6.30.1; Git is also installed. No GPU is available or required, and the task declares no additional dependencies.

The build timeout is 600 seconds and the complete evaluator timeout is 1800 seconds. Within evaluation, the candidate numerical subprocess has a 100-second timeout and each notebook cell has a 60-second timeout. Direct numerical execution fixes PyTorch and BLAS-style thread counts to one despite the two-vCPU container allocation. The notebook uses a non-interactive Matplotlib backend and an installed `python3` IPython kernel. Patch storage counts against the 4,096 MiB workspace limit, and the submitted diff itself is separately limited to 1,048,576 bytes.
