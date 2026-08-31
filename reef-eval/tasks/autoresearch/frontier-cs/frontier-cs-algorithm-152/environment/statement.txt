Problem Statement
--------
AtCoder Inc. operates a food delivery service, AtCoder Foods, that leisurely delivers food that tastes good even if it gets cold.
This service receives a large number of delivery orders in advance, and processes multiple deliveries simultaneously to improve efficiency.
The current service area is represented as a square area $\\{(x,y)\mid 0\leq x, y\leq 800\\}$ on a two-dimensional plane, with AtCoder's office located at the center $(400, 400)$.
There are 1000 orders today, and the $i$ ($1\leq i\leq 1000$)-th order is a food delivery request from a restaurant in $(a_i, b_i)$ to a location in $(c_i, d_i)$.

Today's quota for Takahashi, a delivery man, is to process 50 orders.
He can freely choose a subset $S\subseteq\\{1,\cdots,1000\\}$ of size exactly 50 from the 1000 orders and deliver on a route $(x_1,y_1),\cdots,(x_n,y_n)$ satisfying the following conditions.

1. For each $i\in S$, visit $(c_i, d_i)$ after visiting $(a_i,b_i)$. That is, there exists an integer pair $(s, t)$ such that $(x_s,y_s)=(a_i,b_i)$, $(x_t,y_t)=(c_i,d_i)$, and $s<t$.
2. $(x_1,y_1)=(x_n,y_n)=(400, 400)$.

After picking up food at one restaurant, he may pick up food at another restaurant or deliver food to another destination before delivering that food to the destination.
He is so powerful that he can carry arbitrary numbers of dishes simultaneously.

Moving from $(x_i,y_i)$ to $(x_{i+1},y_{i+1})$ takes time equal to the Manhattan distance $|x_i-x_{i+1}|+|y_i-y_{i+1}|$, and
the total travel time for the delivery route is $T=\sum_{i=1}^{n-1} |x_i - x_{i+1}|+|y_i - y_{i+1}|$.
Please optimize $S$ and delivery routes so that the total travel time is as short as possible.


Scoring
--------
For the total travel time $T$ of the output delivery route, you will get a score of $\mathrm{round}(10^8/(1000+T))$.

There are 100 test cases, and the score of a submission is the total score for each test case. If you get a result other than <span class='label label-success' data-toggle='tooltip' data-placement='top' title="Accepted">AC</span> for one or more test cases, the score of the submission will be zero. The highest score obtained during the contest time will determine the final ranking, and there will be no system test after the contest. If more than one participant gets the same score, the ranking will be determined by the submission time of the submission that received that score.

Input
--------
Input is given from Standard Input in the following format:

~~~
$a_1$ $b_1$ $c_1$ $d_1$
$\vdots$
$a_{1000}$ $b_{1000}$ $c_{1000}$ $d_{1000}$
~~~

Each $a_i, b_i, c_i, d_i$ is an integer between $0$ and $800$, inclusive, where $(a_i, b_i)$ represents the coordinates of the restaurant, and $(c_i, d_i)$ represents the coordinates of the destination.
$(a_i,b_i)\neq (c_i,d_i)$ is satisfied, but for different orders $j$, there is a possibility that $\\{(a_i,b_i),(c_i,d_i)\\}\cap\\{(a_j,b_j),(c_j,d_j)\\}\neq\emptyset$.

Output
--------
Let the set of chosen orders be $r_1,\cdots,r_m$ ($1\leq r_i\leq 1000$), and the delivery route be $(x_1,y_1),\cdots,(x_n,y_n)$ ($0\leq x_i,y_i\leq 800$), output to Standard Output in the following format.

~~~
$m$ $r_1$ $\cdots$ $r_m$
$n$ $x_1$ $y_1$ $\cdots$ $x_n$ $y_n$
~~~

You may output multiple times for visualization purposes.
If your program outputs multiple times, only the last output will be used for scoring.
The final output must satisfy $m=50$, but intermediate outputs with $m\neq 50$ are allowed for visualization.


Input Generation
--------
Let $\mathrm{rand}(L,U)$ be a function that generates a uniform random integer between $L$ and $U$, inclusive.
For each $i=1,\cdots,1000$, we generate an order $(a_i, b_i, c_i, d_i)$ as follows.

We generate $a_i=\mathrm{rand}(0, 800)$, $b_i=\mathrm{rand}(0, 800)$, $c_i=\mathrm{rand}(0, 800)$, and $d_i=\mathrm{rand}(0, 800)$.
Redo the generation as long as the Manhattan distance $|a_i-c_i|+|b_i-d_i|$ is less than 100.


Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc006/c21daebb77aa4d38d65f4d7f7c7249.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
- <a href="https://img.atcoder.jp/ahc006/c21daebb77aa4d38d65f4d7f7c7249.html">Web version</a>: This is more powerful than the local version and can display animations.

**Sharing visualization results is not allowed until the end of the contest. **

{sample example}
