DNA Matching

Time Limit: 2 s
Memory Limit: 512 MB

A string s is called a “DNA sequence” if and only if s consists only of the characters A, C, G, T.

You are given m strings s1, s2, ..., sm, each of length n, and each of them consists only of the characters A, C, G, T, ?.

For a DNA sequence t of length n, we call t valid if and only if there exists an index i (1 ≤ i ≤ m) and a way to replace each ? in si by one of A,C,G,T, so that after replacement the resulting string s_i' is exactly equal to t.

You need to compute the probability that a randomly chosen DNA sequence t (of length n, where each of the 4^n possible DNA sequences is equally likely) is valid.

Input Format
The first line contains two positive integers n, m.
Then follow m lines, each line contains a string si of length n. It is guaranteed that each si only uses characters from A, C, G, T, ?.

Output Format
Output one real number — the probability. Make sure the number is precise enough.

Sample 1
input
3 1
AC?
output
0.0625

Sample 2
input
6 2
AC??A?
A??A?T
output
0.0302734375
