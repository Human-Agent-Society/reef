Graph Coloring

You are given an undirected graph G with n nodes. Each node is colored either black (0) or white (1). You are given the initial coloring state and the target coloring state. Your task is to transform the graph from the initial state to the final state using a sequence of valid transformations.

In one transformation, each node simultaneously changes its color to either:
- Its current color (stays the same), or
- The color of one of its neighbors

Your goal is to minimize the number of transformations needed. You can use at most 20,000 transformations.

Input
The first line contains two integers n (1 ≤ n ≤ 1000) and m (1 ≤ m ≤ 1e5) — the number of nodes and edges in the graph.
The second line contains n integers representing the initial state.
The third line contains n integers representing the target state.
The next m lines each contain two integers u and v, indicating an undirected edge between nodes u and v. The graph has no self-loops or multiple edges.
It is guaranteed that a solution exists

Output
On the first line, print a single integer k — the number of transformation steps in your solution.
The next k+1 lines should each contain n integers representing the coloring state at each step.

Scoring
You will be graded based on the number of transformations you use.
Your answer will be compared to a reference solution ref_steps. Your final score will be calculated as the average of 100 * min((ref_steps + 1) / (your_steps + 1), 1) across all test cases.

Time limit: 2 seconds
Memory limit: 512 MB

Sample Input:
6 6
1 1 0 1 0 1
1 0 0 0 1 0
1 2
2 3
3 5
5 4
4 6
6 5

Sample Output:
2
1 1 0 1 0 1
1 1 0 0 1 0
1 0 0 0 1 0