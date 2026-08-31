Story
--------
Mr. Takahashi, the mayor of Takahashi City, decided to draw a map of Takahashi City on the floor of the city hall lobby using colored square tiles.
Takahashi City is divided into several wards, and in this map, each ward should be represented as a set of connected tiles of the same color.
He commissioned a contractor to create a draft of an accurate map, but the number of tiles to be used was too large, and the budget was exceeded.
Mayor Takahashi, who loves graphs, is only interested in the adjacencies between the wards and thinks that the map could be drawn with fewer tiles if information other than adjacencies, such as the shape and size of each ward, is ignored.
Please create a map using as few tiles as possible.

<div style="display: flex; width: 100%;">
  <div style="flex-basis: 40%; text-align: center; margin-right: 10%;">
    <img src="./images/input.png" style="max-width: 100%; max-height: 100%; vertical-align: middle;">
	<p>Accurate map</p>
  </div>
  <div style="flex-basis: 50%; text-align: center;">
    <img src="./images/output.png" style="max-width: 100%; max-height: 100%; vertical-align: middle;">
	<p>Small map correctly representing adjacencies</p>
  </div>
</div>

Problem Statement
--------
Given a map of Takahashi City represented on a grid of $n\times n$ squares.
Let $(0,0)$ be the coordinates of the top-left square, and $(i,j)$ be the coordinates of the square located $i$ squares down and $j$ squares to the right from there.
The city consists of $m$ wards, and the square of color $c$ ($1\leq c\leq m$) corresponds to the $c$-th ward.
The outside of the $n\times n$ squares correspond to the outside of the city and is colored $0$.

Two squares are defined as "adjacent" if they share an edge, and a set of squares is defined as "connected" if any two squares can reach each other via adjacent squares.
In the given map, for each color c, the set of squares of color c is guaranteed to be connected.

Your task is to create a map represented on a grid of $n\times n$ squares that satisfies all of the following conditions.

- For every color $c$ ($0\leq c\leq m$), squares of color $c$ must be connected. Note that since the outside of the $n\times n$ squares is colored $0$, squares of color $0$ can be connected through the outside squares.
- For every pair of colors $c$ and $d$ ($0\leq c<d\leq m$), the adjacency of a set of squares of color $c$ and a set of squares of color $d$ in the original map and the created map must be identical. That is, if and only if there exist adjacent squares of color $c$ and $d$ in the original map, there exist adjacent squares of color $c$ and $d$ in the created map. Note that since the outside of the $n\times n$ squares is colored $0$, the squares on the boundary are considered to be adjacent to squares of color $0$.


Scoring
--------
Let $E$ be the total number of squares of color $0$ in the created map.
Then you will obtain a score of $E+1$.

There are 150 test cases, and the score of a submission is the total score for each test case.
If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> or <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> , and the score of the submission will be zero.
The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time.



Input
--------
Input is given from Standard Input in the following format.

~~~
$n$ $m$
$c_{0,0}$ $c_{0,1}$ $\cdots$ $c_{0,n-1}$
$\vdots$
$c_{n-1,0}$ $c_{n-1,1}$ $\cdots$ $c_{n-1,n-1}$
~~~

For all test cases, we fix $n = 50$ and $m = 100$.
$c_{i,j}$ is an integer value representing the color of the square at coordinates $(i,j)$ and satisfies $1\leq c_{i,j}\leq m$.
For every $k=1,2,\cdots,m$, there exists at least one $(i,j)$ with $c_{i,j}=k$.


Output
--------
Let $d_{i,j}$ ($0\leq d_{i,j}\leq m$) be the color of the square at coordinates $(i,j)$ in the created map.
Then, output to Standard Output in the following format.

~~~
$d_{0,0}$ $d_{0,1}$ $\cdots$ $d_{0,n-1}$
$\vdots$
$d_{n-1,0}$ $d_{n-1,1}$ $\cdots$ $d_{n-1,n-1}$
~~~

If the output map does not satisfy the conditions specified in the problem statement, the submission will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span>.

Your program may output multiple solutions.
If multiple solutions are output, only the last one is used for scoring.
You can compare multiple solutions using the web version of the visualizer.

<a href="https://img.atcoder.jp/ahc024/AU5KcDyn.html?lang=en&seed=0&output=sample">Show example</a>


Input Generation
--------
<details>
First, we initialize with $c_{i,j}=0$ for all $(i,j)$.
Next, for each $k=1,2,\cdots,m$, we randomly select a square with $c_{i,j}=0$ and set $c_{i,j}=k$.
Finally, we repeat the following process while squares with $c_{i,j}=0$ remain.

Randomly select a square with $c_{i,j}=0$ and randomly select its adjacent square $(i',j')$.
We set $c_{i,j}=c_{i',j'}$.
</details>

Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc024/AU5KcDyn.html?lang=en">Web version</a>: This is more powerful than the local version providing animations and manual play.
- <a href="https://img.atcoder.jp/ahc024/AU5KcDyn.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc024/AU5KcDyn_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

Please be aware that sharing visualization results or discussing solutions/ideas during the contest is prohibited.
