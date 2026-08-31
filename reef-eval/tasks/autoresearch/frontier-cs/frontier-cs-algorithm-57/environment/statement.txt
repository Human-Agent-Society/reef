time limit per test: 5 seconds
memory limit per test: 1024 megabytes
**Submissions are only allowed in C++ language.**


This problem is interactive.

Baudelaire is very rich, so he bought a tree of size n, rooted at some arbitrary node. Additionally,
every node has a value of 1 or -1.

Cow the Nerd saw the tree and fell in love with it. However, computer science doesn't pay him
enough, so he can't afford to buy it. Baudelaire decided to play a game with Cow the Nerd, and if
he won, he would gift him the tree.

Cow the Nerd does not know which node is the root, and he doesn't know the values of the nodes
either. However, he can ask Baudelaire queries of two types:

Type 1 query: Let f(u) be the sum of the values of all nodes in the path from the root of
the tree to node u. Cow the Nerd may choose an integer k and k nodes u1,u2,...,uk,
and he will receive the value f(u1)+f(u2)+...+f(uk).

To ask this query, print:
? 1 k u1 u2 ... uk
and read an integer from the interactor.

Type 2 query: Baudelaire will toggle the value of node u. Specifically, if the value of u is 1, it will
become -1, and vice versa.

To ask this query, print:
? 2 u
and read the response (no output, just toggle).

Cow the Nerd wins if he guesses the value of every node correctly (the values of the final tree,
after performing the queries) within Q total queries. Can you help him win?

Scoring
If your solution makes at most n queries, you will receive 100 points.
If it makes more than n+1000 queries, you will receive 0 points.
If it makes x queries where n < x ≤ n+1000, your score will be linearly interpolated
from 100 down to 0.

Input
The first line of the input contains a single integer t (1 ≤ t ≤ 100), the number of test cases.

The first line of each test case contains a single integer n (2 ≤ n ≤ 1000), the size of the tree.

Each of the next n-1 lines contains two integers u and v (1 ≤ u, v ≤ n, u ≠ v),
denoting an edge between nodes u and v in the tree.

It is guaranteed that the sum of n over all test cases does not exceed 1000 and that each graph
provided is a valid tree.

Interaction
After printing a query do not forget to output the end of line and flush the output. Otherwise, you
may get the Idleness Limit Exceeded verdict.

When you have found the answer, output:
! v1 v2 ... vn
where vi is the value of node i after performing the queries.

Printing the answer does not count as a query.

Example

input
3
4
1 4
4 2
2 3
1
-1
-5
-5
2
1 2
2
7
1 2
2 7
7 3
7 4
7 5
7 6
-1

output
? 1 3 1 2 4
? 1 2 3 1
? 2 4
? 1 3 1 2 4
? 1 2 3 1
! -1 -1 -1 -1
? 1 1 1
! -1 1 1 1 1 1 -1