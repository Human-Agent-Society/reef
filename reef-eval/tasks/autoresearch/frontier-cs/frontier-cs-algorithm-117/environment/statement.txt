Line

The territory of country P is a square with side length 2 * 10^12.  
The origin is placed at the center of the square, and a Cartesian coordinate system is established so that the sides of the square are parallel to the axes.  
Thus, the territory of country P is the region [-10^12, 10^12] × [-10^12, 10^12].

There are N lines on this territory, each of the form y = a_i * x + b_i.  
Both a_i and b_i are unknown integers between -10^4 and 10^4.  
You do not know their values.

Your task is to recover all these N lines.  
To do this, you may ask the king up to Q_max queries of the following type:

- You give a point (x, y), and the king tells you the sum of the distances from (x, y) to all N lines.

You need to recover all the lines by making no more than Q_max queries.

Input

The only line of input contains one integer n.

Implementation details

You may issue queries by writing to standard output lines of the form 

    ? x y

This query sends the interactor a query point (x, y).  
You must ensure that (x, y) is inside the region [-10^12, 10^12] × [-10^12, 10^12], x and y do not have to be integers.
The interactor returns the sum of distances from (x, y) to all N lines.  
You may call this function at most Q_max times.

You must also make a guess exactly once of the form

    ! a_1 a_2 ... a_n b_1 b_2 ... b_n

You may output the lines in any order.

Subtasks and scoring

If your program fails the time limit (1.0 s), memory limit (256 MiB),  
or produces wrong output, the score for that test point is 0.

Otherwise, let Q be the number of queries you made and S be the full score of that test point:

- If Q > Q_max, score = 0.
- If Q_min < Q <= Q_max, score = S * (1 - 0.7 * (Q - Q_min) / (Q_max - Q_min)).
- If Q <= Q_min, score = S.

Constraints:
1 <= N <= 100  
-10^4 <= a_i, b_i <= 10^4  
Q_max = 10^4, Q_min = 402
No two lines are parallel.

Time limit: 1 second  
Memory limit: 256 MB
