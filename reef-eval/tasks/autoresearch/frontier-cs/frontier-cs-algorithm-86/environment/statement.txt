Hidden Tree

This is an interactive problem.

There is a hidden tree with n nodes. The nodes are numbered from 1 to n. Your task is to discover the structure of this tree by asking queries.

Each query consists of three distinct node numbers. The interactor will return the node that minimizes the sum of distances to these three nodes.

Your goal is to determine the structure of the tree. You are allowed to make at most 20,000 queries.

This problem is graded based on the number of queries you use. In order to receive any points, you must use no more than 20,000 queries. Your answer will be compared to a reference solution ref_queries. Your final score will be calculated as the average of 100 * min((ref_queries + 1) / (your_queries + 1), 1) across all test cases.

Input

There is only one test case in each test file.

The first line of the input contains an integer n (3 ≤ n ≤ 1000) indicating the number of nodes in the hidden tree.

Interaction

To ask a query, output one line. First output 0 followed by a space, then print three distinct integers from 1 to n separated by spaces. After flushing your output, your program should read a single integer representing the node that minimizes the sum of distances to the three queried nodes.

When you have determined the tree structure, output one line. First output 1 followed by a space, then print n-1 pairs of integers representing the edges. The format should be: u1 v1 u2 v2 ... u(n-1) v(n-1), where each pair (ui, vi) represents an edge in the tree. All values should be separated by spaces. After flushing your output, your program should exit immediately.

Note that the tree structure for each test case is pre-determined. That is, the interactor is not adaptive.

To flush your output, you can use:
- fflush(stdout) (if you use printf) or cout.flush() (if you use cout) in C and C++.
- System.out.flush() in Java.
- stdout.flush() in Python.

Example

Input:
3

2

Output:

0 1 2 3

1 1 2 2 3

Time Limit: 2 seconds
Memory Limit: 512 MB