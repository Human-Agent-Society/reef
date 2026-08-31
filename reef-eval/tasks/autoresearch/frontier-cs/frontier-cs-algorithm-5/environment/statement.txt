Problem: Hamiltonian Path Challenge

You are given a directed graph with n vertices and m edges.  
Your task is to find a path that visits each vertex exactly once.  

If it is not possible to find a Hamiltonian path, you should instead output a path that is as long as possible and does not repeat vertices.  

----------------------------------------
Input
The first line contains two integers n, m.  
The second line contains 10 integers, where the i-th number is a_i, the scoring parameter.  
Each of the next m lines contains two integers u, v, representing a directed edge from u to v.

----------------------------------------
Output
Output two lines:

- The first line contains an integer k, the number of vertices in your path.  
- The second line contains k integers, the sequence of vertices in the path.

----------------------------------------
Sample 1
Input:
3 3
3 3 3 3 3 3 3 3 3 3
1 2
1 3
2 3

Output:
3
1 2 3

Explanation:
Edges are directed.  
In this case, the contestant’s submission scores 10 points.

----------------------------------------
Sample 2
Input:
4 4
1 1 2 2 3 3 4 4 4 4
1 2
2 1
1 3
4 2

Output:
2
2 1

Explanation:
In this case, the submission scores 4 points.  
Note: you do not need to output the optimal solution.

----------------------------------------
Scoring
- If your output is invalid, your score is 0.  
- Otherwise, let k be the number of vertices in your path.  
  Your score is equal to:  

  sum_{i=1}^{10} [k ≥ a_i]  

That is, the number of a_i values that are less than or equal to your k.

----------------------------------------
Constraints
- 1 ≤ n, m ≤ 500000  
- No multiple edges, no self-loops  
- At least one Hamiltonian path exists in the graph

----------------------------------------
Time Limit: 4 second  
Memory Limit: 512 MB