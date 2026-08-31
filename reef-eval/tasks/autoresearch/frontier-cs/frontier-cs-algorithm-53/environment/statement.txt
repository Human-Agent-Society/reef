Problem: G2. Inter Active (Hard Version)

Time limit: 2 seconds

Memory limit: 512 MB

This is the hard version of the problem. The difference between the versions is that in this version, you can make at most 10 * n queries. You can hack only if you solved all versions of this problem.

Ali loved Bahamin's gift (from problem E) so much that he illegally traveled from Qazvin to Liverpool to have the gift signed by football players. Now Interpol is searching for him, but they've offered a deal: solve a problem, and he can stay in Liverpool. But since he's currently at the stadium, he can't solve it so he asked you to do it.

This is an interactive problem.

There is a hidden permutation p of length n >= 4 where p_i != i for each 1 <= i <= n.

Initially, you should give the jury a positive integer k <= n, which will be constant through future queries.

Then you need to find permutation p using some queries.

In each query, you give a permutation q_1, q_2, ..., q_n to the jury. In response, you will receive the number of pairs (i, j) such that all of the following conditions hold:
- i < j
- p_{q_i} = q_j
- i != k (k is the constant you have given to the jury)

You are given n, and you need to find the permutation p in at most 10 * n queries.

A permutation of length n is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array), and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 <= t <= 500). The description of the test cases follows.

The only line of each test case contains a single integer n (4 <= n <= 100) — the length of p.

It is guaranteed that the sum of n^2 over all test cases does not exceed 10^4.

Interaction Protocol

The interaction for each test case begins with reading the integer n.

Then you should output the integer k (1 <= k <= n). This is not considered as a query.

Then you can ask up to 10 * n queries. To make a query, output a line in the following format:
? q_1 q_2 ... q_n

The jury will return the answer to the query.

When you find the permutation p, output a single line in the following format:
! p_1 p_2 ... p_n

This is also not considered as a query.

After that, proceed to process the next test case or terminate the program if it is the last test case.

The interactor is not adaptive, which means that the permutation is determined before the participant outputs k.

If your program makes more than 10 * n queries, your program should immediately terminate to receive the verdict Wrong answer. Otherwise, you can get an arbitrary verdict because your solution will continue to read from a closed stream.

After printing each query do not forget to output the end of line and flush the output. Otherwise, you will get Idleness limit exceeded verdict.

If, at any interaction step, you read -1 instead of valid data, your solution must exit immediately.

Note:
In the first test case, p=[3,1,4,2]. The solution selected k=1 then it asked permutation q=[1,2,3,4]. Only pair (3,4) satisfies the conditions.
In the second test case, p=[3,1,2,5,4]. The solution selected k=3. For permutation q=[1,2,5,4,3], only pair (1,5) satisfies the conditions. For permutation q=[2,1,4,3,5], pairs (1,2) and (2, 4) satisfy the conditions.

Example Input:
2
4
1

5
1
2

Example Output:
1
? 1 2 3 4
! 3 1 4 2
3
? 1 2 5 4 3
? 2 1 4 3 5
! 3 1 2 5 4