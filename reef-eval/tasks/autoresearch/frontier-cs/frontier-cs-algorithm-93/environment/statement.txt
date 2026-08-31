Greedy

Time limit: 7 seconds
Memory limit: 512 MB
This is an interactive problem. All communication is via standard input/output (no input/output files).

Story
Little U tackles an optimization on trees but only has a greedy “black box.” You must reconstruct the hidden tree using queries to this black box.

Core Problem
You are given a hidden rooted tree with n nodes labeled 1..n. Every node has either 0 children or at least 2 children.
Let A be any set of nodes with no ancestor–descendant pairs (an antichain). The size of the maximum such set depends on the tree.

Greedy Black Box (what your queries measure)
For any sequence (order) of distinct node labels vec (length sz ≥ 1):
• Start with an empty chosen set S.
• Scan vec from left to right. For the current node x:
  – If x is neither an ancestor nor a descendant of any node already in S, insert x into S.
  – Otherwise skip x.
• The black box outputs |S|, the number of chosen nodes.

Your Task
Reconstruct the entire parent array par[1..n] of the tree (par[root] = 0) by interactively querying the black box.

Standard I/O Protocol

Initial input (from the judge)
The first line contains two integers:
  n ty
where:
• n — number of nodes (1 ≤ n ≤ 3000)
• ty — a test identifier chosen by the judge (for your program, it has no algorithmic effect and can be ignored)

Making a query
To ask the black box for the greedy result on a sequence vec of length sz:
  1) Print a line in the following format exactly:
       ? sz v1 v2 ... vsz
     where each vi is a distinct integer in [1, n] and all vi are pairwise distinct.
  2) Immediately flush stdout after printing the line.
  3) Read a single integer R from stdin — this is the black box answer |S| for your sequence.

Submitting the final answer
When you have reconstructed the tree, print a single line:
  ! p1 p2 ... pn
with:
• pi = parent of node i (for all i = 1..n)
• The root’s parent must be 0 (i.e., p_root = 0).
After printing the answer, flush stdout and terminate without issuing further queries.

Protocol Requirements (binding)
• You must flush stdout after every output line you produce (both “?” and “!” lines).
• You must not output any extra text beyond the specified formats; no prompts, logging, or debug prints.
• For each query, you must print exactly one “?” line, flush, then read exactly one integer reply.
• After the “!” line is printed and flushed, your program must exit immediately.
• The judge evaluates only the first “!” line. Any subsequent output will be ignored and may cause a Wrong Answer verdict.
• Each query (“?” line) will be processed in O(n) time on the judge side.
• The sequence in a “?” query can be any subset order (no duplicates required to cover all nodes).

Scoring (0–100 scale based on number of queries)
Let T be the number of queries you print (the number of lines beginning with “?”).
• If T ≤ 45,000 → score = 100
• If T > 200,000 → score = 0
• Otherwise (45,000 < T ≤ 200,000) → use a smooth linear scale strictly inside (0, 100):
    score = clamp(0.01, 99.99, 100 * (200000 - T) / 155000)
  where clamp(a, b, x) = min(b, max(a, x)).
  (This yields 99.99 when T is just above 45,000, and 0.01 when T is 200,000.)

Constraints (single overall setting; no subtasks beyond the original hardest setting)
• n ≤ 3000
• Node labels are 1..n
• Each node has either 0 children or at least 2 children

Sample interaction (illustrative)
Judge → Program:
  3 0
Program → Judge:
  ? 2 2 3
(Program flushes)
Judge → Program:
  2
Program → Judge:
  ! 0 1 1
(Program flushes and exits)

Explanation of the sample
The query “? 2 2 3” returns 2, implying that nodes 2 and 3 can both be taken simultaneously by the greedy, so neither is an ancestor of the other; since every non-leaf has ≥ 2 children and n = 3, this suggests both 2 and 3 are children of 1. The final answer line “! 0 1 1” declares 1 as root (parent 0) and parents of 2 and 3 as 1, which the judge accepts.