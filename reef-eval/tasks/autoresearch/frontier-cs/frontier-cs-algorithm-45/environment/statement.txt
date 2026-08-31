Problem: Balanced Graph Partitioning (DIMACS10-style, EC & CV)

Overview
--------
You are given an undirected, unweighted graph G = (V, E) and an integer k (a power of two).
Assign each vertex to one of k parts so that:
  1) the balance constraint is satisfied, and
  2) two quality measures are minimized with equal importance:
     • Edge Cut (EC): the number of edges whose endpoints are in different parts.
     • Communication Volume (CV): defined below.

This statement precisely defines the graph model, balance rule, metrics, I/O, and scoring.

Definitions
-----------
Graph model
• Input may contain duplicate edges and self-loops; the judge reduces to a simple undirected graph
  by ignoring self-loops and merging parallel edges.
• Vertices are labeled 1..n.

Partition
• A k-way partition is p : V → {1,2,…,k}. Empty parts are allowed.
• k is always a power of two in the official tests.

Balance constraint
• Let n = |V| and k be given. Let ideal = ceil(n / k). Let eps be the slack from the input.
• Every part must satisfy:  size(part) ≤ floor((1 + eps) * ideal).

Edge Cut (EC)
• EC(p) = |{ {u,v} ∈ E : p(u) ≠ p(v) }| (minimize).

Communication Volume (CV)
• For a vertex v with part P = p(v), define F(v) = number of DISTINCT other parts Q ≠ P such that
  v has at least one neighbor in part Q.
• For each part Q, Comm(Q) = Σ_{v : p(v)=Q} F(v).
• CV(p) = max_{Q} Comm(Q) (minimize).

Input
-----
Line 1:  n m k eps
Lines 2..m+1:  u v   (1 ≤ u,v ≤ n).

Output
------
Print exactly n integers, the labels p_1 … p_n with 1 ≤ p_i ≤ k. Whitespace is free.

Scoring
-------
The .ans file accompanying each test provides four integers:
    bestEC bestCV baselineEC baselineCV
Per metric, the checker computes:
    s = clamp((baseline - your) / (baseline - best), 0, 1)    (minimization)
and returns score = (s_EC + s_CV) / 2.

Important for this pack:
• We set bestEC = 0 and bestCV = 0 for all tests. These are theoretical lower bounds (ideal targets) and
  do not imply that EC=0 or CV=0 is achievable on a given graph or with k>1. They are used solely for
  normalization, i.e. your normalized improvement is (baseline - your) / baseline (clamped to [0,1]).
• baselineEC and baselineCV are computed from median performance of multiple random balanced partitions
  (or may be fixed by the organizers).

Constraints
-----------
• Time:   1 s per test
• Memory: 512 MB
• k: power of two
• eps = 0.03 in this dataset
• Graphs are large and structure-rich (R-MAT, BA, SBM-like, regular/expander-ish, torus, 3D grid).

Validation by the checker
-------------------------
• Exactly n labels read; each in [1..k].
• Balance enforced: size(part) ≤ floor((1 + eps) * ceil(n / k)).
• EC and CV computed on the simplified simple graph.
• Partial credit reported with the substring “Ratio: <score>”.
