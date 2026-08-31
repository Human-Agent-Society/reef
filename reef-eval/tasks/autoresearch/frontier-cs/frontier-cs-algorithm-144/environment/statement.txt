Find Median

This is an interactive problem.

There is a hidden permutation p of length n, where n is even.
You are allowed to make queries by choosing a subsequence of indices with even length k (where 4 ≤ k ≤ n). For a chosen subsequence, the interactor will return the two median values.
For a subsequence of even length k, the two medians are defined as the k/2-th and (k/2+1)-th smallest values in that subsequence.

Your goal is to find the index of the two medians in permutation p. This problem is graded based on the number of queries you use. In order to receive any points, you must use no more than 500 queries. After that, your answer will be compared to a solution ref_queries. Your final score will be calculated as the average of 100 * min(ref_queries / your_queries, 1) across all cases.

Input

The first line contains a single integer n (6 ≤ n ≤ 100, n is even) — the length of the hidden permutation.

Interaction

To make a query, output one line in the following format:
0 k x1 x2 ... xk

where:
- k is the length of the subsequence (4 ≤ k ≤ n, k must be even)
- x1, x2, ..., xk are distinct indices between 1 and n

After each query, read a line containing two integers m1 and m2 (1 ≤ m1 < m2 ≤ n) — the two median values of the subsequence.

When you have found the answer, output one line in the following format:
1 i1 i2 - the index of the two medians.

After printing the answer, your program should terminate immediately.

Note: The interactor is non-adaptive. The permutation is fixed before you start querying.

To flush your output, use:
- fflush(stdout) or cout.flush() in C++
- System.out.flush() in Java
- stdout.flush() in Python

Example Interaction

Input:
6

3 4

3 4

2 3

Output:

0 6 1 2 3 4 5 6

0 4 3 6 1 5

0 4 3 6 2 5

1 3 6

Time limit: 4 seconds
Memory limit: 512 MB