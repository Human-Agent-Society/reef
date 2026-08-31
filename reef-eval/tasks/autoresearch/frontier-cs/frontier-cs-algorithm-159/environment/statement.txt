Problem Statement
--------
RectJoin is the following single-player game played with square grid paper and pencil.

![example](./images/example.gif) 

Let $(0, 0)$ be the coordinates of the lower left corner of the grid paper, with the $x$-axis to the right and the $y$-axis to the top.
The coordinates of the upper right corner of the grid paper are $(N-1, N-1)$.
Initially, $M$ dots are placed on grid points, and you can repeat the following operations as many times as possible to place dots and draw rectangles on the grid paper.

In the $i$-th operation, choose a grid point $p_{i,1}$ not containing a dot and three grid points $p_{i,2}, p_{i,3}, p_{i,4}$ containing dots, which satisfy all of the following three conditions.

1. Connecting $p_{i,1} p_{i,2} p_{i,3} p_{i,4}$ in this order forms a rectangle that is parallel to the axis or inclined at 45 degrees.
2. There are no dots other than $p_{i,2}, p_{i,3}, p_{i,4}$ on the perimeter of this rectangle.
3. The perimeter of this rectangle does not share a common segment of positive length with the perimeter of an already drawn rectangle (intersecting at some points is allowed).

For the chosen four points, place a new dot on $p_{i,1}$ and draw the perimeter of the rectangle $p_{i,1} p_{i,2} p_{i,3} p_{i,4}$ on the grid paper.

Let $(c,c)=((N-1)/2,(N-1)/2)$ be the coordinates of the center of the graph paper.
We define the weight of each grid point as $w(x,y)=(x-c)^2 + (y-c)^2 + 1$ using the distance from the center.
Let $S=\sum_{x=0}^{N-1}\sum_{y=0}^{N-1} w(x,y)$ be the sum of the weights of all grid points.
Let $Q$ be the set of coordinates with dots in the final state (including the initially placed dots).
Then you will get the following score.

\\[\mathrm{round}\left(10^6 \cdot\frac{N^2}{M}\cdot\frac{\sum_{(x, y)\in Q} w(x, y)}{S}\right)\\]

Create a program to play the game to get as high a score as possible.

#### Additional explanation of the rules

- You must choose $p_{i,1}$ from the interior of the grid paper, i.e., from coordinates satisfying $0\leq x,y\leq N-1$.
- Since a dot is placed on the grid point chosen as $p_{i,1}$, it cannot be chosen again as $p_{j,1}$ in $j(>i)$-th operation, but it can be chosen again as $p_{j,k} (k=2,3,4)$.
- Similarly, $p_{i,k} (k=2,3,4)$ can be repeatedly chosen as $p_{j,k'} (k'=2,3,4)$ in $j(>i)$-th operation.
- As stated in condition 2, there must be no other dots on the perimeter of the chosen rectangle, but conversely, you can choose a point on the perimeter of an already drawn rectangle as $p_{i,1}$ and place a dot on it.

#### Number of test cases
- Provisional test: 50
- System test: 2000. We will publish <a href="https://img.atcoder.jp/ahc014/seeds.txt">seeds.txt</a> (sha256=907b41fcba240515612a21798a10b0df7dda744b1268b74b3bbd41b93a73095e) after the contest is over.
- System test contains 125 inputs for each of $N=31,33,35,\cdots,61$.
- The input of seed=0 is manually created and is not included in the provisional or system test.

The score of a submission is the total scores for each test case.
In the provisional test, if your submission produces illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> or <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> , and the score of the submission will be zero.
The final ranking will be determined by the system test with more inputs which will be run after the contest is over.
In the system test, if your submission produces illegal output or exceeds the time limit for some test cases, only the score for those test cases will be zero.
The system test will be performed only for <font color="red"><strong>the last submission which received a result other than <span class="label label-warning" data-toggle="tooltip" data-placement="top" title="" data-original-title="Compilation Error">CE</span> </strong></font>.
Be careful not to make a mistake in the final submission.

#### About execution time
Execution time may vary slightly from run to run.
In addition, since system tests simultaneously perform a large number of executions, it has been observed that execution time increases by several percent compared to provisional tests.
For these reasons, submissions that are very close to the time limit may result in <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> in the system test.
Please measure the execution time in your program to terminate the process, or have enough margin in the execution time.


Input
--------
Input is given from Standard Input in the following format:

~~~
$N$ $M$
$x_1$ $y_1$
$\vdots$
$x_M$ $y_M$
~~~

- $N$ is an odd number between 31 and 61, representing the number of vertical and horizontal grid points on the grid paper.
- $M$ denotes the number of initially placed dots, satisfying $N\leq M\leq \lfloor N^2/12 \rfloor$.
- $(x_1, y_1), \cdots, (x_M, y_M)$ denote the coordinates of the $M$ dots, each of which satisfies $\lfloor N/4\rfloor\leq x_i,y_i\leq\lfloor 3N/4\rfloor$.

Output
--------
Let $K$ be the total number of operations and $(x_{i,1}, y_{i,1}), (x_{i,2}, y_{i,2}), (x_{i,3}, y_{i,3}), (x_{i,4}, y_{i,4})$ be the four points chosen in the $i$-th operation.
Then, output to Standard Output in the following format.

~~~
$K$
$x_{1,1}$ $y_{1,1}$ $x_{1,2}$ $y_{1,2}$ $x_{1,3}$ $y_{1,3}$ $x_{1,4}$ $y_{1,4}$
$\vdots$
$x_{K,1}$ $y_{K,1}$ $x_{K,2}$ $y_{K,2}$ $x_{K,3}$ $y_{K,3}$ $x_{K,4}$ $y_{K,4}$
~~~

The order of the four points can be clockwise or counterclockwise, but $(x_{i,1}, y_{i,1})$ must be the point where the new dot is placed.


<a href="https://img.atcoder.jp/ahc014/a3c240f5b1.html?lang=en&seed=0&output=20%0D%0A9+15+12+12+15+15+12+18%0D%0A15+20+12+17+15+14+18+17%0D%0A23+22+19+22+19+12+23+12%0D%0A23+14+22+15+21+14+22+13%0D%0A10+14+10+13+12+13+12+14%0D%0A11+11+12+11+12+12+11+12%0D%0A18+20+15+20+15+19+18+19%0D%0A19+16+22+19+21+20+18+17%0D%0A12+19+12+18+15+18+15+19%0D%0A15+22+12+19+15+16+18+19%0D%0A14+22+15+22+15+24+14+24%0D%0A15+8+18+11+15+14+12+11%0D%0A10+15+9+15+9+14+10+14%0D%0A11+18+12+19+10+21+9+20%0D%0A22+23+20+21+21+20+23+22%0D%0A21+15+18+15+18+14+21+14%0D%0A15+26+13+24+15+22+17+24%0D%0A20+20+16+24+14+22+18+18%0D%0A21+17+18+20+15+17+18+14%0D%0A11+14+10+13+11+12+12+13%0D%0A">Show example</a>


Input Generation
--------
The input of seed=0 is manually created and is not included in the provisional or system test.
Let $\mathrm{rand}(L,U)$ be a function that generates a uniform random integer between $L$ and $U$, inclusive.

#### Generation of $N$ and $M$
$N=\mathrm{rand}(15, 30)\times 2 + 1$,
$M=\mathrm{rand}(N, \lfloor N^2/12 \rfloor)$.

#### Generation of $(x_i, y_i)$
$M$ grid points $(x,y)$ satisfying $\lfloor N/4\rfloor\leq x,y\leq\lfloor 3N/4\rfloor$ are chosen at random.

Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc014/a3c240f5b1.html?lang=en">Web version</a>: This is more powerful than the local version and can display animations and manual play.
- <a href="https://img.atcoder.jp/ahc014/a3c240f5b1.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc014/a3c240f5b1_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

<font color="red">
You are allowed to share output images (PNG) of the provided visualizer for seed=0 on twitter during the contest.
Note that sharing in animation format is prohibited.
</font>
You have to use the specified hashtag and public account.
You can only share visualization results and scores for seed=0.
Do not share GIFs, output itself, scores for other seeds or mention solutions or discussions.

<a href="https://twitter.com/search?q=%23AHC014%20%23visualizer&src=typed_query&f=live">List of shared images</a>

{sample example}