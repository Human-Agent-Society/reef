AveragePermutation

This is an interactive problem.

There is a hidden permutation p of n, where n is even. You want to discover this permutation using queries.
Each query consists of asking about k specific positions a1, a2, ..., ak. The query is answered with 1 if the average of the elements at these positions is an integer, and 0 otherwise.
Note that permutations [p1, p2, ..., pn] and [n+1-p1, n+1-p2, ..., n+1-pn] are indistinguishable using these queries. Therefore, you are guaranteed that p1 ≤ n/2.
This problem is graded based on the number of queries you use. Specifically, your answer will be compared to a reference solution ref_queries.
Your final score will be calculated as the average of 100 * min((ref_queries + 1) / (your_queries + 1), 1) across all test cases.

Input
The first line of the input contains an even integer n (2 ≤ n ≤ 800) indicating the length of the hidden permutation.

Interaction
To ask a query, output one line. First output ? followed by a space, then print an integer k (1 ≤ k ≤ n), then print k distinct integers a1, a2, ..., ak (1 ≤ ai ≤ n) separated by spaces. After flushing your output, your program should read a single integer (0 or 1) indicating whether the average is an integer.
If you want to guess the permutation, output one line. First output ! followed by a space, then print a permutation of n separated by spaces, satisfying the constraint that p1 ≤ n/2. After flushing your output, your program should exit immediately.

Note that the answer for each test case is pre-determined. That is, the interactor is not adaptive. Also note that your guess does not count as a query.

To flush your output, you can use:
    fflush(stdout) (if you use printf) or cout.flush() (if you use cout) in C and C++.
    System.out.flush() in Java.
    stdout.flush() in Python.

Time limit: 4 seconds
Memory Limit: 256 MB

Example input:

? 2 1 2

! 1 3 2

Example output:
3

1