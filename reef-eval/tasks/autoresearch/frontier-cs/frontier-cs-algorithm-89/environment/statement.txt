Problem: Tree Reconstruction via Steiner-Membership Queries

Time limit: 3 seconds
Memory limit: 512 MB

This is an interactive problem.

You are given an unknown unweighted tree T with n labeled vertices V = {1, 2, ..., n}. Your task is to determine all edges of T by interacting with a judge.

Definition
For any subset S ⊆ V, define Steiner(S) as the smallest connected subgraph of T that contains all vertices in S. Equivalently, Steiner(S) is the union of all simple paths between all pairs of vertices in S. For a vertex v ∈ V, we say “v is on Steiner(S)” if v belongs to this subgraph.

Interaction
- At the start, read a single integer n (n <= 1000 in all official tests).
- You may then issue queries of the following form to the judge:

  Print a line:
  ? k v s1 s2 ... sk

  where:
  - k is an integer with 1 ≤ k ≤ n,
  - v is a vertex label with 1 ≤ v ≤ n,
  - s1, s2, ..., sk are k distinct integers in [1, n] forming the set S,
  - v may or may not belong to S; both are allowed.

  After each query, read a single integer from standard input:
  - 1 if v ∈ Steiner(S),
  - 0 otherwise.

- When you have determined the entire tree, output your answer:

  First print a line containing just:
  !

  Then print exactly n − 1 lines, each containing two integers u v (1 ≤ u, v ≤ n), describing the edges of the tree. You may output the edges in any order. After printing all edges, flush and terminate your program.

Important notes
- Flushing: After every query and after the final answer, flush your output. For example:
  - C++: fflush(stdout) or cout << endl << flush;
- Validity: If you print an invalid query (wrong format, out-of-range labels, duplicate elements in S, k = 0, etc.), the judge may terminate your program with a Wrong Answer verdict.
- Non-adaptiveness: The hidden tree is fixed before interaction begins and does not depend on your queries.

Global limit on total set sizes
To ensure the judge runs efficiently, the total size of all sets you submit across all queries must not exceed 3,000,000. Formally, if your queries are (k1, ...), (k2, ...), ..., then sum(ki) over all queries must be ≤ 3,000,000. If you exceed this limit, the judge may terminate the interaction immediately (e.g., by returning -1 or by issuing Wrong Answer). If you ever read -1, you must exit immediately.

Input/Output summary
- Read:
  - n (once at the beginning)
  - After each query: one integer (0/1), or possibly -1 if you violated the protocol (exit immediately if so)
- Write:
  - Queries: ? k v s1 s2 ... sk
  - Final answer:
    !
    u1 v1
    u2 v2
    ...
    u_{n−1} v_{n−1}

Scoring
Your solution is judged by both correctness and the number of queries.

- Let Q be the total number of queries you make.
- Full score (100 points) if Q ≤ 3,000.
- Zero score if Q > 1,200,000.
- Otherwise, your score is interpolated linearly:
  score = 100 × (1,200,000 − Q) / (1,200,000 − 3,000), rounded to the nearest integer.

Only solutions that correctly reconstruct the tree are scored. Violations of the interaction protocol, printing an invalid final tree, or exceeding the total-set-size limit result in 0 points for that test.

Example interaction
This example is illustrative only. The actual judge will use a hidden tree with n = 1000.

Judge: 5
You: ? 2 2 1 3
Judge: 1
You: ? 2 5 1 3
Judge: 0
You: ? 3 4 1 3 5
Judge: 1
You: !
You: 1 2
You: 2 3
You: 2 4
You: 4 5

Additional clarifications
- S must be non-empty (k ≥ 1).
- If |S| = 1 with S = {x}, then Steiner(S) is the single vertex x. A query “? 1 v x” returns 1 if and only if v = x.
- The final answer must describe a simple tree on vertices {1, 2, ..., n}. Edges can be in any order, but duplicates and self-loops are not allowed.