Problem
There are $N$ balls numbered from $1$ to $N$, and three baskets numbered from $1$ to $3$. Initially, all $N$ balls are in basket $1$.

Balls can be moved from one basket to another according to the following rules:

When the balls in a basket are arranged in numerical order, the ball in the middle is called the center ball.
If the number of balls is even, the center ball is the one with the larger number between the two middle balls.
When moving a ball from basket $a$ to basket $b$, the center ball of basket $a$ must be moved to basket $b$, and the moved ball must become the center ball of basket $b$.
Using this rule, output the process of moving all $N$ balls from basket $1$ to basket $3$.

Input
The first line contains the number of balls $N$. ($1 \le N \le 30$)

Output
The first line should output the number of moves $M$. 
The next $M$ lines should each contain two integers $a$ and $b$, separated by a space. ($1 \le a, b \le 3; a \ne b$) This indicates that the $i$-th operation moves a ball from basket $a$ to basket $b$ according to the problem's rules.

After the output, all balls originally in basket $1$ must be moved to basket $3$.