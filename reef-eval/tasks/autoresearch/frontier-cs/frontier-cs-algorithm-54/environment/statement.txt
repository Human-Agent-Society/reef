Problem: Centroid Guess

Time limit: 4 seconds

Memory limit: 512 MB

This is an interactive problem.

There is an unknown tree consisting of n nodes, which has exactly one centroid.
You only know n at first, and your task is to find the centroid of the tree.

You can ask the distance between any two vertices. However, making too many queries will reduce your score, and exceeding 400,000 queries will result in 0 points (and potentially a Wrong Answer verdict).

Note that the interactor is not adaptive. That is, the tree is fixed in each test beforehand and does not depend on your queries.

A vertex is called a centroid if its removal splits the tree into subtrees with at most floor(n/2) vertices each.

Input

The only line of the input contains an integer n (3 <= n <= 7.5 * 10^4) the number of nodes in the tree.

Interaction Protocol

Start interaction by reading n.

To ask a query about the distance between two nodes u, v (1 <= u, v <= n) output "? u v".

If you determine that the centroid of the tree is x, use "! x" to report.

After printing a query, do not forget to output the end of a line and flush the output.
Otherwise, you will get Idleness limit exceeded. To do this, use:
- fflush(stdout) or cout.flush() in C++;
- System.out.flush() in Java;
- flush(output) in Pascal;
- stdout.flush() in Python;
- see documentation for other languages.

Scoring

Your score depends on the number of queries Q you use to identify the centroid.

Let K_base = 100000 and K_zero = 400000.

The score is calculated using the following quadratic formula:
Score = max(0, 100 * ((K_zero - Q) / (K_zero - K_base))^2)

Specifically:
- Base Score (100 pts): If you use Q <= 100000 queries, you will receive at least 100 points.
- Partial Score: If you use between 100000 and 400000 queries, your score will decrease quadratically. For example, using 250000 queries yields 25 points.
- Zero Score: If you use Q >= 400000 queries, you will receive 0 points.
- Bonus Score: This problem supports unbounded scoring. If your solution uses fewer than 100000 queries, your score will follow the same curve and exceed 100 points.

Hacks are disabled in this problem.
It's guaranteed that there are at most 500 tests in this problem.

Example Input:
5

2

1

2

3

1

1

1

Example Output:
? 1 2

? 1 3

? 1 4

? 1 5

? 2 3

? 3 4

? 4 5

! 3