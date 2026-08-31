Ball Moving Game

Xiao C is playing a ball-moving game. In front of him, there are n + 1 pillars, numbered from 1 to n + 1. On each of the first n pillars, there are m balls placed from bottom to top, while pillar (n + 1) initially has no balls. Altogether there are n × m balls in n different colors, with exactly m balls of each color.

At the beginning, the balls on a pillar may be of different colors. Xiao C’s task is to move all balls of the same color onto the same pillar. This is the only objective, and there is no restriction on which pillar each color ends up on.

Xiao C can achieve this by performing a sequence of operations. In each operation, he can move the top ball from one pillar to another. More specifically, moving a ball from pillar x to pillar y must satisfy:

Pillar x has at least one ball.

Pillar y has at most (m - 1) balls.

Only the top ball of pillar x can be moved, and it must be placed on top of pillar y.

The task itself is not too difficult, so Xiao C adds a restriction for himself: the total number of operations must not exceed 10^7. In other words, Xiao C needs to complete the goal using at most 10^7 operations. Specifically, your score will be (10^7 - k) / 10^7, where k is the number of operations.

Although Xiao C feels stuck, he believes you can solve it. Please output a valid sequence of operations to achieve the goal. Multiple correct solutions may exist; you only need to output one. It is guaranteed that at least one valid solution exists.

Input

The first line contains two integers n and m: the number of colors, and the number of balls of each color.

The following n lines each contain m integers, separated by spaces.

For the i-th line, the integers (from bottom to top) give the colors of the balls on pillar i.

Output

The first line of your output should contain a single integer k, the number of operations in your solution, where 0 ≤ k ≤ 10^7.

Each of the following k lines should contain two integers x and y, meaning you move the top ball of pillar x onto the top of pillar y.

You must guarantee that 1 ≤ x, y ≤ n + 1 and x ≠ y.

Sample Input 1

3 2
1 2
3 2
1 2


Sample Output 1

6
2 1
3 2
3 2
3 1
3 2
3 2


Explanation of Sample 1

Pillars are shown as stacks from bottom to top.
After performing the operations step by step, all balls of the same color are gathered on the same pillar.