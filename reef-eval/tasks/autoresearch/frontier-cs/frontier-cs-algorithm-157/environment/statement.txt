Story
--------
Takahashi loves puzzles and is playing with the following famous sliding puzzle.
> There are $N^2-1$ tiles on an $N \times N$ board.
> There is a single empty square, and you can slide an adjacent tile in any of the four directions into the empty square.
> Some picture is divided into each tile. By repeatedly sliding the tiles, please align the picture.

The trouble is, Takahashi had thrown away the instruction manual, so he lost the target picture.
According to his memory, the target picture was a <a href="https://en.wikipedia.org/wiki/Tree_(graph_theory)">tree</a>.
By repeating the sliding operation, please complete a tree.

![example](./images/example.gif)

Problem Statement
--------
There are $N^2-1$ tiles on an $N \times N$ board.
Let $(i, j)$ denote the coordinates of row $i$ $(0\leq i \leq N-1)$ from the top and column $j$ $(0\leq j\leq N-1)$ from the left.
Each tile contains a figure with lines from its center towards one or more of four directions: up, down, left, and right.
We represent each tile using a bitmask with 1 for left, 2 for up, 4 for right, and 8 for down, as follows.

<table>
<tr align="center">
<td>
Tile
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<rect fill="lightgray" height="40" width="40" x="0" y="0"/>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
<td>
<svg height="50" id="vis" viewBox="-5 -5 50 50" width="50" xmlns="http://www.w3.org/2000/svg">
<rect fill="white" height="50" width="50" x="-5" y="-5"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="0" y2="0"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="0" y1="0" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="0" x2="40" y1="40" y2="40"/>
<line stroke="lightgray" stroke-width="1" x1="40" x2="40" y1="0" y2="40"/>
<g transform="translate(0,0)">
<line stroke="#905020" stroke-width="10" x1="20" x2="0" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="0"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="40" y1="20" y2="20"/>
<line stroke="#905020" stroke-width="10" x1="20" x2="20" y1="20" y2="40"/>
<circle cx="20" cy="20" fill="#905020" r="5"/>
</g>
</svg>
</td>
</tr>
<tr align="center">
<td>
Binary
</td>
<td>
0000
</td>
<td>
0001
</td>
<td>
0010
</td>
<td>
0011
</td>
<td>
0100
</td>
<td>
0101
</td>
<td>
0110
</td>
<td>
0111
</td>
<td>
1000
</td>
<td>
1001
</td>
<td>
1010
</td>
<td>
1011
</td>
<td>
1100
</td>
<td>
1101
</td>
<td>
1110
</td>
<td>
1111
</td>
</tr>
<tr align="center">
<td>
Hex
</td>
<td>
0
</td>
<td>
1
</td>
<td>
2
</td>
<td>
3
</td>
<td>
4
</td>
<td>
5
</td>
<td>
6
</td>
<td>
7
</td>
<td>
8
</td>
<td>
9
</td>
<td>
a
</td>
<td>
b
</td>
<td>
c
</td>
<td>
d
</td>
<td>
e
</td>
<td>
f
</td>
</tr>
</table>

The number 0 represents an empty square, and there is exactly one empty square.
With a single operation, you can slide one of the tiles adjacent to the empty square in the four directions to the location of the empty square. After the move, the square from which the tile was moved becomes an empty square.
You can repeat the sliding operation at most $T=2\times N^3$ times.

After finishing the operations, consider a graph with $N^2-1$ squares other than the empty square as vertices and the following edges.

- For each $(i, j)$ $(0\leq i\leq N-2, 0\leq j\leq N-1)$, if $(i,j)$ is a tile with a downward line and $(i+1,j)$ is a tile with an upward line, then construct an edge between $(i,j)$ and $(i+1,j)$.
- For each $(i, j)$ $(0\leq i\leq N-1, 0\leq j\leq N-2)$, if $(i,j)$ is a tile with a rightward line and $(i,j+1)$ is a tile with a leftward line, then construct an edge between $(i,j)$ and $(i,j+1)$.

Your task is to find a short sequence of operations such that the size of the largest tree in this graph, i.e., the number of vertices of the largest connected component without cycles, is as large as possible.
It is guaranteed that within $T$ operations you can construct a tree of size $N^2-1$ with the empty square in $(N-1,N-1)$.
Note that the final position of the empty square is arbitrary and you do not have to move it to $(N-1,N-1)$.

Scoring
--------
Let $K$ be the number of operations and $S$ be the size of the largest tree painted on the board after applying the sequence of operations.
Then, you will get the following score.

- If $S<N^2-1$, $\mathrm{round}\left(500000\times \frac{S}{N^2-1}\right)$
- If $S=N^2-1$, $\mathrm{round}\left(500000\times (2-\frac{K}{T})\right)$

If the number of operations exceeds $T$ or you perform an illegal operation to move a non-existent tile to the empty square, it will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> .

#### Number of test cases
- Provisional test: 50
- System test: 3000. We will publish <a href="https://img.atcoder.jp/ahc011/seeds.txt">seeds.txt</a> (sha256=041256f962c6ba1a60294ad7a575684d6e401163cba316cf978f2e66a4f7b0e3) after the contest is over.
- Both provisional and system tests contain the same number of inputs for each $N=6,7,8,9,10$.

The score of a submission is the total scores for each test case.
In the provisional test, if your submission produces illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> or <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> , and the score of the submission will be zero.
In the system test, if your submission produces illegal output or exceeds the time limit for some test cases, only the score for those test cases will be zero.

#### About execution time
Execution time may vary slightly from run to run.
In addition, since system tests simultaneously perform a large number of executions, it has been observed that execution time increases by several percent compared to provisional tests.
For these reasons, submissions that are very close to the time limit may result in <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> in the s
ystem test.
Please measure the execution time in your program to terminate the process, or have enough margin in the execution time.


Input
--------
Input is given from Standard Input in the following format:

~~~
$N$ $T$
$t_{0,0}$ $\cdots$ $t_{0,N-1}$
$\vdots$
$t_{N-1,0}$ $\cdots$ $t_{N-1,N-1}$
~~~

$N$ is an integer representing the height and width of the board, satisfying $6\leq N\leq 10$.
In all test cases, $T=2\times N^3$.
$t_{i,0}$ $\cdots$ $t_{i,N-1}$ is a string of length $N$.
The $j$-th character $t_{i,j}$ is `0`-`9` or `a`-`f` which is the hexadecimal representation of the figure contained in the tile $(i,j)$.

Output
--------

By representing each operation of sliding the upward, downward, leftward, or rightward adjacent tile into the empty square by a single character `U`, `D`, `L` or `R`, respectively, output the sequence of $K$ operations as a string of length $K$ in one line to Standard Output.

<a href="https://img.atcoder.jp/ahc011/df8bb452a2.html?lang=en&seed=0&output=RRRDLUULDDDDLUUUR">Show example</a>


Input Generation
--------
<details>

#### Generation of $N$ and $T$
We generate $N$ as the remainder of the seed value divided by 5 + 6.
Hence, you can generate inputs with a specific $N$ value by adjusting the seed value.
We set $T=2\times N^3$.

#### Generation of $t_{i,j}$
Let $[k]=\\{0,1,\cdots,k-1\\}$.
We randomly generate a spanning tree $(V,F)$ with vertices $V=[N]\times [N]\setminus \\{(N-1,N-1)\\}$ as follows.

1. First, we randomly shuffle edges $\\{\\{(i,j),(i+1,j)\\}\mid (i,j)\in [N-1]\times [N]\setminus \\{(N-2,N-1)\\}\\}\cup\\{\\{(i,j),(i,j+1)\\}\mid (i,j)\in [N]\times [N-1]\setminus \\{(N-1,N-2)\\}\\}$ and obtain an ordered edge list $e_0, e_1, \cdots$.
2. Starting from $F=\emptyset$, for each $e_k=\\{(i,j),(i',j')\\}$, we insert $e_k$ into $F$ if $(i,j)$ and $(i',j')$ are not connected in $(V,F)$.

From the obtained spanning tree, we construct tiles on which a tree of size $N^2-1$ is drawn, as follows.

1. For each $(i,j)$, if $\\{(i,j),(i+1,j)\\}\in F$, then draw a downward line on tile $(i, j)$ and an upward line on tile $(i+1,j)$.
2. For each $(i,j)$, if $\\{(i,j),(i,j+1)\\}\in F$, then draw a rightward line on tile $(i, j)$ and a leftward line on tile $(i,j+1)$.

Finally, starting from the constructed tile layout, randomly perform $T=2\times N^3$ sliding operations, and let $t$ be the tile layout after the operations.
Here, the $k (\geq 2)$-th operation is chosen uniformly at random from at most three directions excluding the direction that reverts the $(k-1)$-th operation.

</details>

Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc011/df8bb452a2.html?lang=en">Web version</a>: This is more powerful than the local version and can display animations.
- <a href="https://img.atcoder.jp/ahc011/df8bb452a2.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc011/df8bb452a2_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

<font color="red">
You are allowed to share output images (PNG) of the provided visualizer for seed=0 on twitter during the contest.
Note that sharing in video format is prohibited.
</font>
You have to use the specified hashtag and public account.
You can only share visualization results and scores for seed=0.
Do not share GIFs, output itself, scores for other seeds or mention solutions or discussions.
<font color="red">
(Added) The visualizer has a feature to change the value of N, but sharing visualization results for changed inputs is also prohibited.
</font>

<a href="https://twitter.com/search?q=%23AHC011%20%23visualizer&src=typed_query&f=live">List of shared images</a>

{sample example}
