Story
--------
To solve the shortage of police officers, the Takahashi City Police Department has decided to introduce automated patrols with unmanned patrol cars.
The unmanned patrol car is equipped with a high-resolution omnidirectional camera on the roof, which can see the entire road at once in a straight line from the current position. And then, it uses image processing technology to automatically detect suspicious activities.
In order to provide a safe and secure life for the citizens, we want to set up a patrol route that allows the patrol car to see every corner of the city at least once.
Among such patrol routes, please find as short a one as possible.

Problem Statement
--------
You are given a map consisting of $N\times N$ squares.
Let $(0,0)$ denote the top-left square, and $(i,j)$ denote the square at the $i$-th row from the top and $j$-th column from the left.
Each square is either an obstacle (`#`) or a road, and you can move up, down, left, or right on the road squares.
Each road square contains a number `5`-`9`, which represents the amount of time you take to move from an adjacent square to that square.
We define that a road square $(i',j')$ is visible from $(i,j)$ if and only if the following conditions are satisfied:

- $i=i'$ and for every $j''$ with $\min(j,j')\leq j''\leq\max(j,j')$, $(i,j'')$ is a road square, or
- $j=j'$ and for every $i''$ with $\min(i,i')\leq i''\leq\max(i,i')$, $(i'',j)$ is a road square.

For example, in the figure below, the gray squares represent obstacles, the white and light yellow squares represent roads, and the road squares that are visible from the green circle are colored light yellow.

![](./images/1bc7c896310a65486d0ce3aa275f41b7.png "Example of visible squares")

Your task is to find a route starting from a specified square $(si,sj)$, moving up, down, left, or right on road squares, and returning to $(si,sj)$, such that all the road squares become visible at least once.
The shorter the route, the higher the score.
You can move on the same square multiple times and even make a U-turn.

Scoring
--------
Let $r$ be the total number of road squares, $v$ be the number of road squares that become visible at least once, and $t$ be the total travel time of the output route. Then you will obtain the following score.

- If $v<r$, $\mathrm{round}(10^4\times \frac{v}{r})$.
- If $v=r$, $\mathrm{round}(10^4+10^7\times \frac{N}{t})$.

If the output is illegal (going out of $N\times N$ squares, moving on obstacle squares, or not returning to $(si,sj)$), it will be judged as `WA`.
There are 100 test cases, and the score of a submission is the total score for each test case. If you get a result other than `AC` for one or more test cases, the score of the submission will be zero. The highest score obtained during the contest time will determine the final ranking, and there will be no system test after the contest. If more than one participant gets the same score, the ranking will be determined by the submission time of the submission that received that score.

Input
--------
Input is given from Standard Input in the following format:

~~~
$N$ $si$ $sj$
$c_0$
$\vdots$
$c_{N-1}$
~~~

- $N$ is an odd integer between $49$ and $69$, inclusive.
- $si, sj$ are integers satisfying $0\leq si\leq N-1$ and $0\leq sj\leq N-1$.
- Each $c_i$ is a string of length exactly $N$ consisting of characters `5`, `6`, `7`, `8`, `9`, and `#`. The $j$-th character ($0\leq j\leq N-1$) represents the square $(i,j)$ as follows.
	- `#` means that the square contains an obstacle. It is guaranteed that $(si, sj)$ does not contain obstacles.
	- `5`-`9` show that the square is a road, and the number represents the amount of time you take to move from an adjacent square to that square.

Output
--------
Let `U`, `D`, `L`, and `R` represent the movement from $(i, j)$ to $(i-1,j)$, $(i+1,j)$, $(i,j-1)$, and $(i,j+1)$, respectively.
Output a string representing the route to Standard Output in one line.

Input Generation
--------
Let $\mathrm{rand}(L,U)$ be a function that generates a uniformly random integer between $L$ and $U$, inclusive.
We first generate an odd integer $N=\mathrm{rand}(25, 35)\times 2 - 1$ which represents the size of the map and a parameter $K=\mathrm{rand}(2 N, 4 N)$ which represents the number of roads.
Starting from an initial map where all squares are obstacles, we repeat the following procedure $K$ times.

1. Generate an integer $d=\mathrm{rand}(0, 1)$ representing the direction of the road.
1. Generate two integers $i=\mathrm{rand}(0, (N-1)/2)\times 2$ and $j=\mathrm{rand}(0, N-1)$ representing the center of the road.
1. Generate an integer $h=\mathrm{rand}(3, 10)$ representing the length of the road.
1. Generate an integer $w=\mathrm{rand}(5, 9)$ representing the travel time.
1. For each $k$ with $\max(j-h,0)\leq k\leq \min(j+h,N-1)$, we overwrite square $(i,k)$ when $d=0$ or square $(k,i)$ when $d=1$ with a road square with travel time $w$.

After the repetition, we keep only the largest connected component of road squares and replace the rest with obstacles.
Finally, we select $(si,sj)$ uniformly at random from the road squares.

Tools
--------
- <a href="https://img.atcoder.jp/ahc005/c746dac8cc11fd18c68063546997666e.zip">Inputs</a>: A set of 100 inputs (seed 0-99) for local testing, including the sample input (seed 0). These inputs are different from the actual test cases.
- <a href="https://img.atcoder.jp/ahc005/dc9ed10f037e2dd4b48ca255dbd470d9.html">Visualizer on the web</a>
- <a href="https://img.atcoder.jp/ahc005/dc9ed10f037e2dd4b48ca255dbd470d9.zip">Input generator and visualizer</a>: If you want to use more inputs, or if you want to visualize your output locally, you can use this program. You need a compilation environment of <a href="https://www.rust-lang.org/ja">Rust language</a>.

{sample example}
