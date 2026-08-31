Given (1 <= n <= 1e2) and (B = 1e15), n integers a_1 … a_n (0 <= a_i <= B) drawn from either (normal, uniform, pareto, exponential) distributions, find a subset of a_1..a_n that sums as close as possible to T=x_i*a_i, x_i drawn from Bernoulli (1/2).

Score = 100 * (15 - log(error + 1)) / 15

25% of the test cases will be from U(0, B)
25% of the test cases will be from N(B/2, B/6)
25% of the test cases will be from Exp(B/2)
25% of the test cases will be from TruncatedPareto(m=B/3, alpha=2, max=B)

Input: 
n T
A_1 a_2 a_3 a_4 .. a_n



Output: 
Print a binary string of length n, denoting the subset selection.

Sample input: 
3 4
1 2 3

Sample output:
101
