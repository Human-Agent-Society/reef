Problem: Distinct Pairwise XOR Set

Time Limit: 1 second
Memory Limit: 512 MB

Description
Given an integer n, find a subset S ⊆ {1, 2, ..., n} such that:
1) For all pairs (a, b) with a, b ∈ S and a < b, the values (a XOR b) are all distinct (i.e., no two different unordered pairs produce the same XOR).
2) |S| ≥ floor(sqrt(n / 2)).

Input
A single integer n (1 ≤ n ≤ 10^7).

Output
- First line: an integer m — the size of the set S.
- Second line: m distinct integers in the range [1, n] — the elements of S, in any order.

Notes
- Any valid S is accepted. You do NOT need to maximize m; you only need m ≥ floor(sqrt(n/2)).
- The pairwise XOR distinctness means the set {a_i XOR a_j | 1 ≤ i < j ≤ m} has size m*(m-1)/2.
- Multiple correct outputs may exist for the same n.
- Print out the sequence with the longest length.

Sample
Input
49
Output
4
1 2 3 4
