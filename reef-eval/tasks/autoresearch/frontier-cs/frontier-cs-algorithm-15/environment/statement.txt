Problem Statement

You are given a sequence p of length n, which is a permutation of the numbers 1, 2, ..., n.

Your goal is to make the permutation lexicographically as small as possible by performing a specific operation at most 4n times, and then minimize the number of operations needed to reach that. 
More specifically, you must do operations to get the lexicographically smallest permutation that's possible after 4n operations, and then you will be scored based on the number of operations needed to reach it.
Your final score will be calculated as the average of 100 * clamp((4 * n - your_operations)/(4 * n - best_operations), 0, 1) across all cases

The Operation

You can cut the sequence into three consecutive non-empty parts and swap the first part with the last part.

Formally, you select two integers x and y that represent the lengths of the prefix and suffix, respectively. These integers must satisfy:

    x > 0

    y > 0

    x + y < n

This splits the sequence into [Prefix | Middle | Suffix]. The operation transforms the sequence to [Suffix | Middle | Prefix].

Input

The first line contains an integer n, the length of the permutation.
The second line contains n space-separated integers, p_1, p_2, ..., p_n.

Output

On the first line, print an integer m, the total number of operations you performed.
On the following m lines, print the two integers x and y you chose for each operation, separated by a space.

Constraints

    3 <= n <= 1000

    The input sequence p is guaranteed to be a valid permutation.