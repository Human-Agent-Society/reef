Ball Game

You are given n+1 poles numbered from 1 to n+1. Initially, poles 1 through n each contain m balls stacked vertically, while pole n+1 is empty. There are n*m balls in total, with n different colors, where each color appears exactly m times.

Your task is to rearrange the balls so that all balls of the same color are on the same pole. The final distribution of colors to poles does not matter, as long as each pole contains balls of at most one color.

You can perform operations to move balls between poles. In one operation, you can move the topmost ball from pole x to the top of pole y, subject to the following constraints:
- Pole x must have at least one ball
- Pole y must have at most m-1 balls

Your goal is to minimize the number of operations needed. You must use at most 2,000,000 operations.

Input

The first line contains two integers n and m (2 ≤ n ≤ 50, 2 ≤ m ≤ 400) — the number of colors and the number of balls of each color.

The next n lines each contain m integers. The i-th line describes the color of balls on pole i from bottom to top. (colors are numbered from 1 to n).

Output

On the first line, print a single integer k (0 ≤ k ≤ 2,000,000) — the number of operations in your solution.

The next k lines should each contain two integers x and y (1 ≤ x, y ≤ n+1, x ≠ y), indicating that you move the topmost ball from pole x to pole y.

It is guaranteed that a valid solution exists.

Scoring

You will be graded based on the number of operations you use.
In order to receive any points, you must use no more than 2,000,000 operations.
After that, your answer will be compared to a reference solution ref_ops. Your final score will be calculated as the average of 100 * min((ref_ops + 1) / (your_ops + 1), 1) across all test cases.

Time limit: 4 seconds
Memory limit: 512 MB

Sample Input:
2 3
1 1 2
2 1 2

Sample Output:
6
1 3
2 3
2 3
3 1
3 2
3 2