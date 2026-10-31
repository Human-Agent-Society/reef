<!-- From FrontierCS 2 (https://github.com/FrontierCS/FrontierCS-2, CC BY 4.0),
task `join_tree_rewriting`; the statement after the first section is unchanged. -->

# Candidate format

The candidate is the complete new text of `2phase_nsa/binary_plan/rewrite_policy.py`. The judge writes it over
that file in the task repository, makes the task's patch, and scores it with the
task's own FrontierCS 2 verifier on the development (train) suite. Every other
file of the repository stays unchanged; this run does not submit helper files.
The verifier message returns the train score and the development metrics that a
FrontierCS 2 agent receives, nothing else.

# Join-Tree Rewriting

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
This task is built from a research artifact for instance-optimal processing of acyclic joins in column stores. The artifact takes binary equijoin trees produced by a query optimizer, translates them into a two-phase nested-semijoin representation, and executes the resulting physical plans in a single-threaded columnar engine. A binary tree identifies the participating relation aliases and the equivalence classes of columns used by each join.

The program reads one optimizer-produced JSON plan at a time. The evaluated plans are drawn from three benchmarks — Stats-CEB (derived from Stack Overflow data), the Join Order Benchmark over IMDB, and a cardinality-estimation graph benchmark — and each plan joins five to fourteen aliases over the 95 Parquet relations the judge ships. It presents the topmost binary join subtree to Python as `BinaryJoinNode` and `LeafNode` objects. The program returns an in-memory binary tree, which the fixed pipeline validates and lowers to a physical nested-semijoin plan. A correct return value contains the same annotated relation leaves and equijoin predicates as the input, satisfies the lowerability invariant specified below, and produces exactly the same query result as the original binary plan. As the secondary objective, maximise the arithmetic mean of the per-query scores from fresh execution speedups over the original binary plans. Reference PASS is a separate conjunction of exact correctness and strict latency improvement on every query.

The `rewrite_plan` policy that supplies the returned tree has been removed and must be implemented.

## Design space
The editable scope is the in-memory transformation performed by `rewrite_plan`. A submission may replace the stub's body and may add private functions, classes, type aliases, and constants in the same module. It may return any tree composed of the shipped `BinaryJoinNode` and `LeafNode` types that meets the output contract; the judge does not require the returned objects to preserve the input objects' identities. Standard output and standard error from the call are discarded, and neither is an output channel.

The function receives one complete join tree at a time. There is no task API for changing scans, filters, projections, equivalence-class construction, physical-plan lowering, or execution. Input parsing, the public node classes, workload JSON, Parquet relations, conversion programs, execution engine, correctness comparison, timing collection, and score calculation are fixed. The source-policy rules in Validity also constrain the module's imports and operations. Only the literal file listed under Your workspace is included in the submitted change.

## Workload
Three benchmarks supply the plans, and they differ in how much a two-phase rewrite
is worth on them. The judge ships all of their relations as Zstandard-compressed
Parquet — 95 files, 468 MiB in total.

| benchmark | public/train plans | hidden plans added in final | relation leaves |
| --- | ---: | ---: | ---: |
| Stats-CEB | 6 | 6 | 5–6 |
| Join Order Benchmark (IMDB) | 5 | 4 | 9–14 |
| Cardinality-estimation graph benchmark | 9 | 9 | 6–8 |

Every plan is an acyclic equijoin of uniquely aliased relation leaves and
initially violates the lowering invariant. Plans retain the scan projections,
filters, aliases, and estimated cardinalities produced by the artifact's
optimizer-to-IR pipeline. The public and hidden partitions are stratified by join size and share no query.
Train uses all 20 public plans. Final freshly evaluates those same 20 public
plans plus all 19 hidden plans: 39 required queries, with no additional queries.
Each workload identity includes its partition, because numeric filenames repeat
between partitions. Cyclic joins, Cartesian
products, non-equality joins, and engine changes are outside this workload.
Suite membership, case count, data scale, and repetition count are fixed rather
than varied by the graded run.

Every lowerable query first executes the original binary plan, the optimized reference, and the candidate once, in that order, to compare exact query results. The timing pass repeats the block BinaryA, ReferenceA, CandidateA, CandidateB, ReferenceB, BinaryB five times. This retains the reference/candidate A-B-B-A order and gives ten fresh observations for each role and query. Each observation times execution and result collection; plan construction, input loading and result formatting are outside that elapsed interval. Query results are also compared against the original binary result in every timing repetition.

## Interface
The input files are JSON physical-plan trees, one file per query, with a top-level `execution_time` number, `root` operator record, and `aliases` object mapping query-local aliases to base relations. The loader starts at `root`, follows any unary operators to the topmost `HASHJOIN`, constructs attribute equivalence classes from every join `condition`, and converts that join subtree into `BinaryJoinNode` and `LeafNode` objects. This complete two-input join record occurs inside a graded workload file:

```json
{
  "name": "HASHJOIN",
  "execution_time": 0.002087,
  "actual_rows": 41569,
  "estimated_cardinality": null,
  "children": [
    {
      "name": "SEQUENTIALSCAN",
      "execution_time": 0.000058,
      "actual_rows": 40324,
      "estimated_cardinality": 40325,
      "children": [],
      "relation": "users",
      "opt_filter": "Id<=55746 AND Id IS NOT NULL",
      "projection": [
        {"table_name": "u", "field_name": "Id"}
      ]
    },
    {
      "name": "SEQUENTIALSCAN",
      "execution_time": 0.000751,
      "actual_rows": 42291,
      "estimated_cardinality": 18396,
      "children": [],
      "relation": "posts",
      "opt_filter": "PostTypeId=1 AND PostTypeId IS NOT NULL AND Score>=-1 AND Score IS NOT NULL AND CommentCount>=0 AND CommentCount<=11 AND CommentCount IS NOT NULL",
      "projection": [
        {"table_name": "p", "field_name": "Id"},
        {"table_name": "p", "field_name": "OwnerUserId"}
      ]
    }
  ],
  "condition": [
    [
      {"table_name": "u", "field_name": "Id"},
      {"table_name": "p", "field_name": "OwnerUserId"}
    ]
  ]
}
```

Here `name` identifies the operator. A scan's `relation` names the Parquet relation, `projection` gives its output fields and their query-local aliases, `opt_filter` is its predicate or `null`, and its `children` list is empty. `actual_rows` records the measured row count, while `estimated_cardinality` supplies the estimate retained in a leaf; a join may have a null estimate. A join's two `children` are full nested operator records, and every two-field entry in `condition` equates one field from each side. Equivalence classes receive integer IDs; a scan becomes `LeafNode(relation_name, attributes, ec)`, where `relation_name` is its alias, `attributes` is the set of equivalence-class IDs present in its projected schema, and `ec` is its estimated cardinality. A join becomes `BinaryJoinNode(left_child, right_child, join_attributes)`.

The judge imports exactly one entry point and calls it once for each query:

```python
def rewrite_plan(plan: BinaryJoinNode | LeafNode) -> BinaryJoinNode | LeafNode
```

The function returns the rewritten tree in memory to the unchanged lowerer; it does not write a file or print a result. A leaf stores `relation_name: str`, `attributes: set[int]`, and `ec: int`. An interior node stores `left_child`, `right_child`, and `join_attributes: set[int]`; its output schema is the union of its children's attributes. For example:

```python
U = LeafNode("u", {1}, 40325)
P = LeafNode("p", {0, 1}, 18396)
plan = BinaryJoinNode(U, P, {1})

out = rewrite_plan(plan)
# One contract-valid result for this one-join input:
out == BinaryJoinNode(U, P, {1})
```

In this example, `u` and `p` are aliases, `{0, 1}` is the set of equivalence classes present in `p`, class 1 is the join predicate, and `40325` and `18396` are estimated cardinalities. After validation, the fixed lowerer writes one `<query-id>.json` physical plan into an evaluator-created output directory. That JSON replaces the original topmost `HASHJOIN` subtree with a `YANNAKAKIS` record whose `root` is a nested `MULTISEMIJOIN`; each such record contains `guard`, `children`, and `equijoin_keys`, and each child is a `GROUPBY` record containing `group_on` and `child`. These files are internal evaluator outputs, not submission artifacts.

For every interior node `P = BinaryJoinNode(P1, P2, J)`, let `LA(Pi)` be the attribute set of the left-most leaf below `Pi`. The required lowerability invariant is:

```
J ⊆ LA(P1)  and  J ⊆ LA(P2)
```

The return value must contain each original relation alias exactly once with exactly the same attribute set and estimated cardinality. It must contain exactly `number_of_leaves - 1` joins, and each returned join's attributes must equal the intersection of the attributes below its left and right children. These requirements preserve the original equijoin predicates and all scan annotations used by lowering. The engine must produce the same query result for the rewritten plan, the comparison plan, and the original binary plan.

Internally, plan construction is invoked with every flag shown here (angle-bracketed values are evaluator-created paths):

```text
<evaluator-Python-interpreter> -I <plan-builder> --app <patched-app> --workloads <suite-directory> --output <plan-output-directory> --upstream <upstream-tools-directory>
```

The execution engine is invoked as:

```text
join_runner --configs <correctness-config-directory> --data <parquet-directory> --timings-outfile <correctness.csv> --repetitions 1 --validation-outfile <correctness.jsonl>
join_runner --configs <timing-config-directory> --data <parquet-directory> --timings-outfile <timings.csv> --repetitions 5 --validation-outfile <timing.jsonl>
```

## Validity
The patch must be a text diff of at most 65,536 bytes, change at most one file, and touch only the listed editable file. An empty patch is accepted by the patch parser. The editable module must be valid Python, define `rewrite_plan`, and contain imports only from `typing` and the shipped `binary_plan` package. Calls to `open`, `exec`, `eval`, `compile`, `input`, `breakpoint`, `__import__`, `globals`, `locals`, `vars`, `getattr`, `setattr`, or `delattr` are forbidden, as is access to any attribute whose name begins with `__`. The source text must not contain `/proc`, `/sys`, `frontier_submission`, `plan_builder`, or `evaluator.py`.

For every case, the returned tree must satisfy every output invariant above and the unchanged conversion pipeline must accept it. A failed or timed-out candidate rewrite contributes zero for that query, fails its PASS comparisons, and stays in the full suite denominator. A leftover output file from a failed rewrite is not accepted. Incorrect candidate query results also score zero for the affected query. No query may be dropped to improve either score or PASS.

Missing, invalid or non-lowerable reference plans are evaluator configuration errors; they never reduce the denominator or become a candidate PASS. Invalid binary/reference measurements, missing engine audit records, and unattributed engine failures are evaluation errors. The engine deadline is cumulative across the phase: any engine timeout is an evaluation error, because its active method alone cannot attribute the accumulated cost to the candidate. The 30-second plan budget applies separately to each query; each of its two conversion subprocesses has a 20-second limit.

## Scoring and reference PASS

For each required query `q`, use this evaluation's ten observations per role:

```
B_q = median(BinaryA and BinaryB execution durations)
C_q = median(CandidateA and CandidateB execution durations)
R_q = median(ReferenceA and ReferenceB execution durations)
S_q = B_q / C_q

L(S_q) = 0,                                      if S_q <= 1
         100 * log(S_q) / (log(10) + log(S_q)),   otherwise

score = sum(L(S_q) for all required q) / N
```

Here `N=20` for train and `N=39` for final. A missing, invalid or incompletely measured candidate query contributes zero with the same fixed divisor. Each scored query must produce the exact original query result. The score uses an arithmetic mean of per-query scores, not a geometric mean of gains. It is zero at or below the original binary baseline, is 50 at a tenfold speedup on each query, and improves smoothly without finite upper clipping. Scores and medians are not rounded. Historical calibration records and the optimized reference do not normalize continuous score.

Every query is one workload with two required PASS comparisons:

1. **Exact result:** the candidate produces exactly the original binary query result in the correctness pass and all timed observations.
2. **Median latency:** `C_q < R_q`, strictly, against the reference measured in this same evaluation. There is no added tolerance. Byte-identical candidate/reference physical plans fail this latency comparison even if timing noise makes the candidate median smaller.

Overall PASS requires both comparisons for every required query. A faster query cannot compensate for a regression, tie, incorrect result, failed rewrite or missing measurement on another query. The continuous score and the reference verdict are separate: a positive or improved score does not imply PASS. An identical reference plan cannot pass the strict speed requirement.

## Calibration and baseline

The optimized reference remains the original checked-in rewrite algorithm. The baseline for continuous scoring is the optimizer's original binary physical plan executed afresh; it is not a rewritten identity policy or a historical timing multiplier. The editable identity stub may fail source policy or lowerability and is not substituted for this binary execution baseline.

## Feedback and grading

Train uses only the 20 public queries. It returns the continuous score, the submission tool's remaining-call information, and aggregate progress on the two PASS comparisons:

- `beat_spec_progress.exact_result` and `beat_spec_progress.median_latency`: how many of the 20 train queries pass and fail that comparison, as `passed`, `failed` and `total` counts with `pass_percent` and `fail_percent`.
- `all_metrics_progress`: the same fields for queries that pass both comparisons. All 20 must pass both for PASS on the train suite.
- `candidate_median_latency_us`: the median over the 20 train queries of `C_q`, your own per-query median latency in microseconds. It is `null` unless every train query was measured.

A rejected patch or a query whose rewrite fails counts as failing both comparisons. The counts come from this evaluation's fresh measurements, so a query whose latency is close to the reference's can pass in one train call and fail in the next. Train feedback names no query and returns no per-query decision, raw duration, reference value, ratio or margin; those are retained only in operator records. Final freshly measures the complete 39-query union, never reusing train measurements, and returns its per-query/per-metric verdict privately. The exact same archive layout is used for train, saved final and explicit final.

## Resources
The container provides 2 vCPUs, 8 GiB of memory, 1 GiB of storage, Python 3.11, Git, the reproduced join executor, and the fixed Stats-CEB Parquet data. Independent final evaluation has a 900-second timeout, while each train verification retains its 240-second ceiling; the build timeout is 120 seconds. Each plan-construction process has a 30-second timeout; its two conversion subprocesses each have a 20-second timeout. Each cumulative engine process has a 600-second timeout. These limits apply to the complete submitted workload, including validation and timing. The resource and timeout values are challenge constants, not values from the source artifact.
