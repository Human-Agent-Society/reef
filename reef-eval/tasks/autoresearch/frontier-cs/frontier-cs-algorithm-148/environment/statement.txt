Problem Statement
--------
There is a floor consisting of $50\times 50$ squares.
The floor is covered with rectangular tiles without any gaps.
Each tile has a size of either $1\times 1$, $1\times 2$, or $2\times 1$ squares.
Let $(0, 0)$ denote the top-left square, and $(i, j)$ denote the square at the $i$-th row from the top and $j$-th column from the left.
Takahashi starts from $(si, sj)$ and walks along a path satisfying the following conditions.

- From $(i, j)$, he can move to $(i-1,j)$, $(i+1,j)$, $(i,j-1)$, or $(i,j+1)$ in one step.
- He can step on the same tile only once. The tile at the initial position is assumed to have already been stepped on.

Each square has an integer value, and the score of a path is the sum of the values of the visited squares, including the square at the initial position.
Your goal is to find a path with as high a score as possible.

Examples
--------
<img src="./images/out1.svg" width=200>
<img src="./images/out2.svg" width=200>
<img src="./images/out3.svg" width=200>

Of the above three figures, only the path in the left figure satisfies the conditions.
In the middle figure, the same tile is stepped on twice in a row.
In the right figure, he left a tile once and then came back to the same tile.

<img src="./images/out.min.svg" width=1002>

Visualization result of the sample output.
The red circle represents the initial position, and the green circle represents the final position.
The tiles stepped on are painted in light blue.

Scoring
--------
The score of the output path is the score for the test case.
If the output does not satisfy the conditions, it is judged as `WA`.
There are 100 test cases, and the score of a submission is the total score for each test case.
If you get a result other than `AC` for one or more test cases, the score of the submission will be zero.
The highest score obtained during the contest time will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, the ranking will be determined by the submission time of the submission that received that score.


Input
--------
Input is given from Standard Input in the following format:

~~~
$si$ $sj$
$t_{0,0}$ $t_{0,1}$ $\ldots$ $t_{0,49}$
$\vdots$
$t_{49,0}$ $t_{49,1}$ $\ldots$ $t_{49,49}$
$p_{0,0}$ $p_{0,1}$ $\ldots$ $p_{0,49}$
$\vdots$
$p_{49,0}$ $p_{49,1}$ $\ldots$ $p_{49,49}$
~~~

- $(si,sj)$ denotes the initial position and satisfies $0\leq si,sj\leq 49$.
- $t_{i,j}$ is an integer representing the tile placed on $(i,j)$. $(i,j)$ and $(i',j')$ are covered by the same tile if and only if $t_{i,j}=t_{i',j'}$ holds. Let the total number of tiles be $M$, then $0\leq t_{i,j}\leq M-1$ is satisfied.
- $p_{i,j}$ is an integer value satisfying $0\leq p_{i,j}\leq 99$ which represents the score obtained when visiting $(i,j)$.

Output
--------
Let `U`, `D`, `L`, and `R` represent the movement from $(i,j)$ to $(i-1,j)$, $(i+1,j)$, $(i,j-1)$, and $(i,j+1)$, respectively.
Output a string representing a path in one line.

Input Generation
--------
#### Generation of $si,sj$
Generate an integer between $0$ and $49$ uniformly at random.

#### Generation of $t_{i,j}$
We start from an initial configuration where tiles of size $1\times 1$ are placed on all squares.
We shuffle the 50x50 squares in random order and perform the following process for each square in order.

- If the tile placed on the current square is $1\times 1$, we randomly select one of the adjacent squares whose tile is $1\times 1$ and connect the two tiles into one tile. If there are no such adjacent squares, we do nothing.
- If the tile placed on the current square is not $1\times 1$, we do nothing.

#### Generation of $p_{i,j}$
Generate an integer between $0$ and $99$ uniformly at random independently for each square.

Tools
--------
- <a href="https://img.atcoder.jp/ahc002/8c847d8177acc2dd417be4327252e39e.zip">Inputs</a>: A set of 100 inputs (seed 0-99) for local testing, including the sample input (seed 0). These inputs are different from the actual test case.
- <a href="https://img.atcoder.jp/ahc002/e5b2b399792299b5b35543c219e89601.html">Visualizer on the web</a>
- <a href="https://img.atcoder.jp/ahc002/c993bb7f09d9f8857fc90951fc6af11d.zip">Input generator and visualizer</a>: If you want to use more inputs, or if you want to visualize your output locally, you can use this program. You need a compilation environment of <a href="https://www.rust-lang.org/ja">Rust language</a>.

{sample example}
