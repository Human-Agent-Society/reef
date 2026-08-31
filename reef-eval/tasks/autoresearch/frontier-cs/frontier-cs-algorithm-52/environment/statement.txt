Geemu

This is an interactive problem.

Little A has an interesting game.
Initially, there is a permutation p of length n. You don't know the permutation p, and you need to determine the final p through the following operations:
1. Query the number of value-contiguous segments in a given interval.
2. Swap two elements at given positions.

Smart little B discovered that no matter what you do, there are always two permutations p that cannot be distinguished. For convenience, you only need to find one of the possible permutations p.
Let s_1 be the number of times you use operation 1, and s_2 be the number of times you use operation 2.

Input
There is only one test case in each test file.
The first line of the input contains three integers n, l_1, and l_2 (1 ≤ n ≤ 10^3, 1 ≤ l_1, l_2 ≤ 10^5) indicating the length of the hidden permutation, the maximum allowed number of ask operations, and the maximum allowed number of swap operations.

Interaction
To query the number of value-contiguous segments in an interval [l, r], output one line:
    1 l r
where 1 ≤ l ≤ r ≤ n.
After flushing your output, read one integer x indicating the number of value-contiguous segments in the interval [l, r].
To swap two elements at positions i and j, output one line:
    2 i j
where 1 ≤ i, j ≤ n.
After flushing your output, read the integer 1 to confirm the swap was performed.
To submit your final answer, output one line:
    3 p_1 p_2 ... p_n
where p_1, p_2, ..., p_n is your determined permutation.
After submitting your report, your program should exit immediately.
To flush your output, you can use:
    fflush(stdout) (if you use printf) or cout.flush() (if you use cout) in C and C++.
    System.out.flush() in Java.
    stdout.flush() in Python.

Scoring
Your solution will be scored based on the efficiency of your queries and swaps, with limits l_1 for operation 1 and l_2 for operation 2.
Let s_1 be the number of queries (operation 1) you use, and s_2 be the number of swaps (operation 2) you use.
Let r_1 be the number of queries (operation 1) the reference solution uses, and r_2 be the number of swaps (operation 2) the reference solution uses.
Your score for each test case is calculated as follows:
1. If s_1 > l_1 or s_2 > l_2, you receive 0 points for that test case.
2. Otherwise, your score for this test case is calculated as:
   score = 100 * min((r_1 + r_2 + 1) / (s_1 + s_2 + 1), 1)
Your final score is the average of your scores across all test cases.

Time limit: 2 seconds
Memoriy limit: 512 MB

Example input:

1 1 2

1 1 3

1 2 3

1 1 3

1 2 4

1 1 4

1 2 5

1 3 5

1 4 5

1 2 5

1 1 5

3 3 5 4 1 2 

Example output:
5 100 50

2

1

1

1

2

2

2

2

1

2

1