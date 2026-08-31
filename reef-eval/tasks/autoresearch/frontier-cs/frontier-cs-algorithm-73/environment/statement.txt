Inversion

Description

This is an interactive problem.

There is a hidden permutation p1, p2, ..., pn of {1, 2, ..., n}.
You want to find it by asking for the parity (even or odd) of the number of inversions in a subarray pl, ..., pr.

You can query in the format:
0 l r
and the interactor will respond with
(sum over all l ≤ i < j ≤ r of [pi > pj]) mod 2

Here, [pi > pj] equals 1 if pi > pj, and 0 otherwise.

Interaction Protocol

1. First, read an integer n (1 ≤ n ≤ 2000).
2. You can make no more than 1999000 queries.

To make a query, output
0 l r
(where 1 ≤ l ≤ r ≤ n)
on a separate line, then read the response from standard input.

To give your final answer, print
1 p1 p2 ... pn
on a separate line.
The output of your final answer does not count toward the limit of 1999000 queries. Your score will be (exp(-q/249875) - exp(-8)) / (1 - exp(-8)), where q is the number of queries.

After printing a query, remember to flush the output, using for example:
- fflush(stdout) or cout.flush() in C++
- System.out.flush() in Java
- flush(output) in Pascal
- stdout.flush() in Python

It is guaranteed that the permutation is fixed in advance.

Example
Standard input:
3
0
0
1

Standard output:
? 1 2
? 1 3
? 2 3
! 2 3 1
