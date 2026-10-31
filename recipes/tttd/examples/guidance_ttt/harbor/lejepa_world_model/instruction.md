<!-- From FrontierCS 2 (https://github.com/FrontierCS/FrontierCS-2, CC BY 4.0),
task `lejepa_world_model`; the statement after the first section is unchanged. -->

# Candidate format

The candidate is the complete new text of `experiments/lejepa_id/losses.py`. The judge writes it over
that file in the task repository, makes the task's patch, and scores it with the
task's own FrontierCS 2 verifier on the development (train) suite. Every other
file of the repository stays unchanged; this run does not submit helper files.
The verifier message returns the train score and the development metrics that a
FrontierCS 2 agent receives, nothing else.

# Latent Objective Recovery

## Problem
This is a self-supervised representation-learning system: it trains an encoder to describe hidden factors behind observations without using those factors as training labels. In the scored experiment, a two-dimensional Gaussian latent state is transformed nonlinearly into an observed point, and two independently perturbed states derived from the same latent row form a positive pair. It comes from the LeJEPA Identifiability research codebase for controlled world-model experiments and retains that project's Gaussian two-dimensional spiral experiment. The program reads a YAML run record and tensors generated online during the run; there is no stored dataset. It trains an encoder whose output is a two-dimensional representation, evaluates that encoder against a fixed 10,000-row latent sample, and writes a JSON result record plus a PyTorch result file. Recovery is measured by how accurately affine maps translate between the latent and learned coordinates in both directions, while orthogonality error measures whether the fitted latent-to-representation linear map preserves Euclidean geometry. A correct run must complete the retained training and evaluation path, emit both valid files with finite scored metrics, and satisfy the loss-interface requirements stated below. The objective is to maximize a score rewarding bidirectional affine recovery, orthogonal-map quality, and lower total wall time across three training workloads.

The bodies of the four loss interfaces in `experiments/lejepa_id/losses.py` have been removed; the scored path requires working implementations of `SIGReg`, `whitening_loss`, and `alignment_loss`, while the retained but unexercised `infonce_loss` interface remains in the editable file.

## Design space
The editable surface is the representation-loss layer called by the fixed training engine, together with Python support code under `experiments/lejepa_id/frontier_solution/`. The loss definitions and their private Python support code are in scope. The encoder architecture, optimizer, learning-rate schedule, generated samples, view construction, nonlinear transformation, training loop, evaluation labels, affine fits, result writer, and metric calculations are fixed. Loss callables receive representation tensors; they are not passed latent coordinates, observations, the encoder, result records, or evaluation metrics.

If you create new Python helpers under `experiments/lejepa_id/frontier_solution/`,
run the following from the working repository before generating your submission:

```sh
git add -N -- experiments/lejepa_id/frontier_solution/
```

This marks the new files for inclusion in `git diff HEAD` without committing
them. A plain diff omits untracked files. Inspect your generated patch to confirm
that it contains every new helper file your loss implementation imports.

## Workload
The development suite and graded suite each contain exactly three fixed-seed runs. They use disjoint seeds drawn from the same Gaussian two-dimensional spiral process. There is no external dataset: every optimizer update generates a fresh batch of 256 independent latent rows `z ~ N(0, I_2)`. For each row, the generator independently samples noise for two views and computes

```text
z_view[v] = 0.95 z + sqrt(1 - 0.95^2) eta[v]
eta[v] ~ N(0, I_2), independently for v in {0, 1}.
```

Consequently, `z_batch.shape == (256, 2)`, `z_view.shape == (2, 256, 2)`, and the two views share the same original latent row but have independent noise. Each view has correlation `0.95` with the original row, and the two generated views have correlation `0.95^2` with one another. The observation transformation is

```text
x = g(z_view) = R(pi ||z_view||_2) z_view,
```

where `R(theta)` is the two-dimensional rotation matrix. The fixed encoder maps these observations to the loss input `h.shape == (2, 256, 2)`.

The scored portion of the generated YAML record has this field layout and these values (the retained file also lists run variants that the command does not select):

```yaml
experiment: 2d
out: <temporary-results-directory>
N: 2
source_dist: gaussian
num_eval: 10000
steps: 1000
lr: 0.003
batch_size: 256
rho: 0.95
log_every: 250
runs:
  spiral_lejepa:
    mixing: spiral
    encoder: mlp
    hidden: 256
    mode: lejepa
    lamb: 0.001
```

The MLP uses width 256, GELU activations, and five `Linear` modules. AdamW uses its PyTorch-default weight decay. The learning rate stays at `0.003` for the first half of the loop and follows a cosine decay during the second half. Although the configuration says `steps: 1000`, the retained loop executes steps 0 through 1000 inclusive, for 1,001 optimizer updates. Thus each run consumes 256,256 freshly generated base latent rows and 512,512 perturbed training-view rows. The original research configuration uses 20,000 steps and a log interval of 500; the scored challenge changes those two fields to 1,000 and 250.

Each run creates one fixed evaluation tensor containing 10,000 Gaussian rows before training. Metrics are evaluated on that tensor 13 times during training and once again after training. After final metrics, the command also generates 100,000 Gaussian rows and writes their latent, observed, and encoded arrays to the PyTorch result. All of this work occurs inside the measured command. The workload is a representative stochastic training workload, not a set of input values to recognize; suite membership and seeds are not disclosed.

## Interface
For each workload, after establishing the sandbox described under `## Validity`, the judge gives the retained runner exactly these arguments from `/workspace/repo/experiments`:

```bash
python3 run.py --config <generated-workload.yaml> --run spiral_lejepa --seed <suite-seed>
```

No other runner flags are passed: in particular, `--N`, `--lamb`, `--rho`, `--alpha`, and `--mode` are absent. The generated YAML is the retained `configs/2d.yaml` with `out` replaced by a temporary results directory, `steps` set to 1,000, and `log_every` set to 250.

The editable file retains these entry points:

- `SIGReg.__init__(knots=17, n_slices=256, t_max=3.0)` must construct a callable PyTorch module. Training uses these defaults; preflight also constructs it with `knots=9`, `n_slices=13`, and `t_max=2.0`.
- `SIGReg.forward(h)` accepts a floating tensor with layout `(views, batch, features)` and must return one scalar tensor.
- `whitening_loss(h)` and `alignment_loss(h)` have the same tensor-to-scalar contract.
- `infonce_loss(h, sigma)` retains the same tensor input plus scalar `sigma`, but the selected `mode=lejepa` workload and preflight do not call it.

This concrete item shows how a latent row becomes the two-view record delivered to the encoder; `h` occupies the same two-view, two-feature layout after encoding:

```text
latent z                    = [ 0.500000, -1.000000]  # (features=2,)
view noise eta[0]           = [ 0.250000,  0.750000]
view noise eta[1]           = [-0.500000,  0.100000]
perturbed latent z_view[0]  = [ 0.553062, -0.715813]
perturbed latent z_view[1]  = [ 0.318875, -0.918775]
h[:, item, :]               = [[h_000, h_001],        # view 0 encoder outputs
                                [h_100, h_101]]        # view 1 encoder outputs
```

For example, every checked callable must satisfy this observable output contract:

```python
>>> h = torch.tensor([[[0.0, 1.0], [1.0, 0.0]],
...                   [[0.2, 0.8], [0.9, 0.1]]], requires_grad=True)
>>> value = SIGReg(knots=9, n_slices=13, t_max=2.0)(h)
>>> value.shape, value.dtype, value.device.type
(torch.Size([]), torch.float32, 'cpu')
>>> value.backward(); h.grad.shape
torch.Size([2, 2, 2])
```

For seed `7`, the runner writes `<generated-out>/spiral_lejepa_seed=7.json` and a same-stem `.pt` file. A JSON result record has this labelled layout; the final three values below illustrate the record shape and are not target values:

```json
{
  "experiment": "2d",
  "run_name": "spiral_lejepa",
  "mixing": "spiral",
  "encoder": "mlp",
  "mode": "lejepa",
  "source_dist": "gaussian",
  "seed": 7,
  "N": 2,
  "lamb": 0.001,
  "rho": 0.95,
  "lr": 0.003,
  "steps": 1000,
  "batch_size": 256,
  "r2_zh": 0.91,
  "r2_hz": 0.90,
  "orth_err_normalized": 0.20,
  "epsilon": 0.45,
  "delta": 0.006,
  "procrustes_mse": 0.10
}
```

In full, the JSON has identity and run fields `experiment`, `run_name`, `mixing`, `encoder`, `mode`, `source_dist`, `source_alpha`, `seed`, `N`, `lamb`, `rho`, `lr`, `steps`, `batch_size`, `n_layers`, and `hidden`; final metric fields `r2_zx`, `r2_xz`, `r2_zh`, `r2_hz`, `orth_err`, `orth_err_normalized`, `epsilon`, `delta`, `D_bound`, `approx_bound`, `procrustes_mse`, `L_h`, and `trace_cov`; the same four recovery fields with a `_grid` suffix; final loss fields `final_align`, `final_sigreg`, `final_whiten`, and `final_loss`; and `log`, whose values are arrays indexed by evaluation point. The `.pt` file contains all result fields, 100,000-row arrays named `z`, `x`, and `h`, and `model_state_dict`. The evaluator reads only `r2_zh`, `r2_hz`, `orth_err_normalized`, `epsilon`, `delta`, and `procrustes_mse` from JSON, requires them to be finite and the last three to be nonnegative, and measures wall time around the whole command rather than using a result field. It also requires the same-stem `.pt` file to be a nontrivial valid PyTorch zip archive containing `data.pkl`; it checks the archive structure without loading candidate-controlled pickle data.

## Validity
The submitted patch must be nonempty text, apply cleanly, touch no more than 20 files, contain no more than 262,144 bytes, and change only the two editable paths. Binary patch content is rejected. Every submitted Python file inspected by the policy check must parse successfully.

Inspected solution files may use only these import forms: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `import math`, and symbol imports from Python modules inside `lejepa_id.frontier_solution`. The external modules expose this exact module-level API to submitted source:

```text
torch:
  Tensor Size device dtype finfo iinfo pi
  float16 bfloat16 float32 float64 int32 int64 bool
  tensor as_tensor arange linspace logspace
  zeros ones empty full zeros_like ones_like empty_like full_like
  rand randn randint randperm rand_like randn_like eye
  cat concat stack vstack hstack split chunk unbind
  abs square pow sqrt rsqrt exp expm1 log log1p sin cos tan tanh
  sigmoid erf erfc softmax logsumexp sum mean var std cov amin amax
  min max minimum maximum clamp where isfinite isnan isinf all any
  sort argsort topk unique matmul mm bmm mv dot inner outer einsum
  triu tril diag diagonal trace kron cdist pdist
  no_grad enable_grad is_grad_enabled manual_seed
torch.nn or nn: Module Parameter ParameterList ModuleList Identity
torch.nn.functional or F:
  normalize mse_loss l1_loss cosine_similarity softmax log_softmax
  relu gelu silu tanh sigmoid pad
torch.linalg:
  norm vector_norm matrix_norm qr svd svdvals solve lstsq inv pinv
  cholesky eig eigh eigvals eigvalsh det slogdet matrix_rank matrix_exp
  cond cross householder_product
torch.special:
  erf erfc erfcx expit gammaln log_ndtr ndtr entr xlog1py xlogy sinc
  softmax log_softmax
torch.autograd: Function grad
math:
  pi e tau inf nan sqrt exp expm1 log log1p sin cos tan tanh erf erfc
  gamma lgamma floor ceil fabs isfinite isnan isinf prod
```

Calls by bare name are limited to solution-defined functions and classes, imported helper symbols, and these built-ins: `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `int`, `isinstance`, `len`, `list`, `map`, `max`, `min`, `next`, `pow`, `range`, `reversed`, `round`, `set`, `sorted`, `sum`, `super`, `tuple`, `zip`, `RuntimeError`, `ValueError`, `TypeError`, and `NotImplementedError`. Tensor and solution-object methods are permitted except that attributes beginning with `__` other than `__init__` and these capability attributes are rejected:

```text
os sys subprocess builtins importlib serialization modules module
system popen fork forkpty vfork spawn _exit
execv execve execl execlp execvp
open fdopen listdir scandir walk chdir getcwd
environ getenv putenv
remove unlink rename replace mkdir makedirs rmdir removedirs
read_text write_text read_bytes write_bytes tofile dump load save
gi_frame gi_code gi_yieldfrom cr_frame cr_code cr_await
ag_frame ag_code ag_await f_globals f_locals f_builtins
f_back f_code f_trace tb_frame tb_next mro
```

The names `open`, `eval`, `exec`, `compile`, `__import__`, `breakpoint`, `getattr`, `setattr`, `delattr`, `vars`, `globals`, `locals`, `input`, `exit`, `quit`, `help`, `license`, `credits`, `__builtins__`, `os`, `sys`, `subprocess`, `importlib`, `builtins`, `SystemExit`, `KeyboardInterrupt`, `type`, and `object` are also rejected. `global` and `nonlocal` declarations are not allowed. Imported names may not be rebound or shadowed. Imported framework objects may not be mutated through assignments, unpacking, loops, comprehensions, context-manager targets, or deletion. Framework modules and submodules cannot be aliased, passed as values, or relayed through helper imports. Framework types may be used directly as constructors, class bases, or direct annotations. Custom metaclasses, class keywords, base-construction hooks, and binding interpreter-managed namespace entries are not allowed. Local helper symbols must be imported directly rather than accessed through an imported helper module; relative imports must stay inside `lejepa_id.frontier_solution`. Helper files must be Python source, may not be symlinks, and are checked under the same policy.

Every preflight and training command executes submitted code in a separate unprivileged process. The runner restricts filesystem access before initializing Torch, then installs a system-call filter across its threads before importing submitted code. The filter denies child-process creation and execution, network sockets, tracing, and cross-process memory access. Submitted source is readable, dedicated scratch and output directories are writable, and judge-owned evaluator and comparator files are outside the filesystem allowlist. These controls work at the verifier's ordinary unprivileged UID and are enforced in addition to the source policy.

Before training, the judge separately evaluates `SIGReg(...)(h)`, `whitening_loss(h)`, and `alignment_loss(h)` on fresh `(2, 32, 2)` `float32` tensors that require gradients. Each call must return a finite rank-zero tensor on the input dtype and device. Backpropagation through each result must give the input a finite, nonzero gradient. This preflight has a 30-second timeout.

Each scored loss (`alignment_loss`, `SIGReg` and `whitening_loss`) must be a pure function of its arguments: apart from the random numbers it draws from Torch's global generator during a call, its value and gradient may not depend on earlier calls, so no call counter, running average, cache or other state may be carried between calls (values fixed at import or in `SIGReg.__init__` are fine). The engine passes a loss no step, epoch or learning rate, and the judge checks this before training: on the same `(2, 256, 2)` input and generator seed, a freshly imported loss and one already called for up to 1,001 optimizer updates must return the same value and input gradient to a relative `1e-4`. This check has the 150-second limit of a training subprocess and counts toward the complete evaluator limit.

Patch-policy failures, build or syntax failures, failed preflight checks, a loss that carries state between calls, crashes, a missing or malformed JSON or `.pt` result, missing or non-finite scored metrics, and a training subprocess exceeding 150 seconds receive a score of zero before performance is considered. An early process exit is therefore not an authentic completed run and receives zero.

## Scoring
The goal is to maximize the score in `[0, 100]`.

For an input matrix `x` and target matrix `y`, the fixed metric appends a column of ones to `x`, obtains `y_hat` by least squares, and computes

```text
R^2(x -> y) = 1 - sum((y - y_hat)^2) / sum((y - mean(y, axis=0))^2).
```

For workload `w`, let `a_w = clip(r2_zh_w, 0, 1)` for the affine fit from latent `z` to representation `h`, and let `b_w = clip(r2_hz_w, 0, 1)` for the reverse fit. For the first fit, write `h_hat = A z + b` and define

```text
o_w = ||A^T A - I_2||_F / sqrt(2).
```

With `clip(x, l, u) = min(u, max(l, x))`, `W = 3` workloads, candidate total command time `T_c = sum_w seconds_c,w`, and withheld-comparator total command time `T_h = sum_w seconds_h,w`, the evaluator computes exactly

```text
Q_w        = 35 a_w + 35 b_w + 20 clip(1 - o_w, 0, 1)
efficiency = min(10, 5 T_h / max(T_c, 1e-9))
score      = round(clip((1/W) sum_w Q_w + efficiency, 0, 100), 4)
```

The weights, clipping bounds, `1e-9` denominator guard, 10-point efficiency component, and four-decimal score resolution are challenge constants (not from materials). The three recovery metrics and the separate per-workload criterion described below use raw, unrounded values.

## Calibration
A naive complete implementation that satisfies the interfaces and trains through the unchanged runner scored `49.7282` in each of three repeated development evaluations, an observed spread of zero. Across its three workloads, `r2_zh` ranged from `0.0058` to `0.9536`, `r2_hz` from `0.3056` to `0.6585`, and normalized orthogonality error from `0.8556` to `0.9949`. These are measurements of the naive implementation, not withheld-comparator values. On the measured development environment, treat a repeatable increase of at least one score point as a genuine improvement rather than timing noise.

## Feedback and grading
Train verification runs asynchronously; retrieve its score with `frontier-submit status <verification_id>`. Stored finals do not run verification. Independent final results are produced after the Agent exits. The submission budget is in `/frontier/task/README.md`. **The graded suite is not the suite you receive feedback on**, and neither its membership nor its seeds are disclosed.

Development feedback gives the arithmetic mean of the candidate's absolute `r2_zh`, `r2_hz`, `orth_err_normalized`, `epsilon`, `delta`, and `procrustes_mse` values across the three development workloads. It does not expose ordered arrays, workload identities, keyed per-workload rows, withheld values, or graded-suite measurements. Separate criterion-progress feedback reports, for each criterion metric, only anonymous passed, failed, total, pass-percent, and fail-percent aggregates; it never reports the workload rows themselves. This progress object is separate from the six candidate-metric means.

A development submission rejected by the loss-purity check scores zero and returns no measurements: its `metrics` instead holds `candidate_diagnostic`, an object with `category` `stateful_loss` and `message`, the fixed sentence of that category. Other rejections score zero without a diagnostic, and final evaluation returns no diagnostic.

A separate stricter criterion compares six final quantities independently on every graded workload: bidirectional affine-recovery R-squared, normalized orthogonality error, and the paper's Appendix H.5 covariance deviation `epsilon` (`||Cov(h) - I||_F`), alignment gap `delta` (`max(L_h - 2(1-rho) tr(Cov h), 0)`), and orthogonal recovery error `procrustes_mse` (mean squared error of `h` against the best orthogonal map of `z`), all computed by the unchanged `lejepa_id/metrics.py` on the same final evaluation set. It is satisfied only if, on every workload, both candidate R-squared values are strictly higher than the corresponding withheld values; the candidate normalized orthogonality error, `epsilon`, and `procrustes_mse` are strictly lower; and the candidate `delta` is strictly lower, except that exact equality at `delta`'s clamp floor of 0 also passes. There is no other tolerance. All six comparisons must hold on all three workloads; better R-squared cannot compensate for a worse bound or recovery quantity. This criterion does not replace the score objective, and a submission that does not satisfy it is still scored normally. The development and graded suites are stratified to carry the same distribution and share no workload.

## Resources
The Agent container provides 2 vCPUs, 12,288 MiB of memory, and 4,096 MiB of workspace storage. Train and final evaluation run in a separate verifier container with 2 vCPUs and 2,048 MiB of memory. The environment is Python 3 with CPU-only PyTorch 2.7.1, NumPy 2.2.6, and PyYAML 6.0.2. Torch and BLAS execution use one thread during scored runs. The build timeout is 900 seconds, the complete evaluator timeout is 400 seconds, the loss-interface preflight timeout is 30 seconds, and each training subprocess has a 150-second timeout. The complete evaluator limit includes interface preflight, the three reference workloads and, for a non-reference submission, the three candidate workloads; the total limit applies even if no individual subprocess reaches its own limit.
