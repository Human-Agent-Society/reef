Time limit:1 second   Memory limit:1024 megabytes

Bobo is given a tree T = (V, E) of n vertices, where there is a number pi on vertex i initially, and p1, p2,..., pn is a permutation of 1 to n, meaning that all integers from 1 to n appear exactly once in p1, p2, ..., pn.
In each operation, Bobo can select a matching M (M belongs to E) (M is a matching means that no two edges in M share a common vertex), and for each (u,v) belongs to M, swap the number on vertex u and vertex v (i.e. swap pu and pv).
Bobo wants to make pi = i for each 1<=i<=n with as few operations as possible, can you please help him?

Input
There are multiple test cases. The first line of the input contains an integer T (T>=1), indicating the number of test cases. For each test case:
- The first line contains a single integer n (10<n<=1000) the number of vertices of the tree.
- The second line contains n integers p1, p2, ..., pn (1<=pi<=n, and p is a permutation of 1 to n), meaning the initial number on vertex i is pi.
Then follow n-1 lines, each with integers u, v (1<= u,v <=n, and u not equals to v) meaning that there is an edge between u and v.
It is guaranteed that the sum of n^2 of all test cases will not exceed 10^6.

Output
For each test case: The first line contains a single integer m (m>=0) meaning the number of operations you used.
Then m lines follow, where each line starts with an integer 0<=ki<n, denoting the number of edges in the matching you select in the i-th operation. Then ki integers t_{i,1}, t_{i,2}, ..., t_{i,ki} follow, denoting the indexes of edges you select.

Scoring
For each test case i (1<=i<=T), your score s_i = max(0, (base_value-m)/(base_value-best_value))
your score on each subtask is 0 if any subtask failed to get the correct result, or the average of all s_i

Example
standard input
1
5
1 4 2 5 3
1 2
2 3
2 4
1 5
standard output
4
2 4 3
1 1
1 2
1 4