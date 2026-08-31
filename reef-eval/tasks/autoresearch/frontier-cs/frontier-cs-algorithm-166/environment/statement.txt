Story
--------
AtCoder frequently hosts onsite contests and has decided to build its own contest venue.
The planned construction site is in a mountain area, and the ground must first be leveled.
Leveling will involve using a dump truck, which incurs costs for loading, unloading, and transporting soil.
Determine the method to level the ground with as little cost as possible.

Problem Statement
--------
There is an $N \times N$ grid of land.
Let $(0,0)$ be the coordinates of the top-left square and $(i,j)$ be the coordinates of the square located $i$ squares down and $j$ squares to the right from there.

The height $h_{i,j}$ of each square $(i,j)$ is given as input.
The total height of all squares is exactly $0$.

Initially, the dump truck is at the square $(0,0)$ with an empty load.
In each turn, you can perform one of the following three operations:

- Specify an integer $d$ satisfying $0<d\leq 10^6$ to load $d$ units of soil from the current square onto the dump truck. This operation decreases the height of the current square by $d$ and increases the amount of soil loaded onto the dump truck by $d$. The height can become negative. This operation incurs a cost of $d$.
- Specify an integer $d$ satisfying $0<d\leq 10^6$ to unload $d$ units of soil from the dump truck to the current square. This operation increases the height of the current square by $d$ and decreases the amount of soil loaded onto the dump truck by $d$. The value of $d$ must not exceed the amount of soil currently loaded onto the dump truck. This operation incurs a cost of $d$.
- Move the dump truck to an adjacent square (up, down, left, or right). The dump truck cannot move outside the $N \times N$ grid. If the dump truck has $d$ units of soil loaded, this operation incurs a cost of $100+d$.

You can perform a maximum of $100000$ turns of operation.
Your task is to determine the sequence of operations that minimizes the cost to make the height of every square $0$.

Scoring
--------
Let $\mathrm{cost}$ be the total cost of the output operation sequence.
Let $\mathrm{base}$ be the sum of $|h_{i,j}|$ for all $(i,j)$.
Let $h_{i,j}'$ be the final height of square $(i,j)$.
Define $\mathrm{diff}$ as the sum of $100|h_{i,j}'|+10000$ for all $(i,j)$ such that $h_{i,j}' \neq 0$.
Then, you will obtain the following score.

\\[
\mathrm{round}\left(10^9\times \frac{\mathrm{base}}{\mathrm{cost}+\mathrm{diff}}\right)
\\]

There are $150$ test cases, and the score of a submission is the total score for each test case.
If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> or <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> , and the score of the submission will be zero.
The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time.



Input
--------
Input is given from Standard Input in the following format:
~~~
$N$
$h_{0,0}$ $\cdots$ $h_{0,N-1}$
$\vdots$
$h_{N-1,0}$ $\cdots$ $h_{N-1,N-1}$
~~~
In all test cases, $N$ is fixed at $20$.
The initial height $h_{i,j}$ of square $(i,j)$ is an integer that satisfies $-100 \leq h_{i,j} \leq 100$, and the total sum of all heights is guaranteed to be $0$.


Output
--------
Let $T$ be the number of operation turns.
The operation in the $t$-th turn is represented by the string $s_t$ as follows:

- The operation to load $d$ units of soil from the current square onto the dump truck: `+d`
- The operation to unload $d$ units of soil from the dump truck to the current square: `-d`
- The operation to move the dump truck to an adjacent square: `U`, `D`, `L`, and `R` for up, down, left, and right, respectively.

Then, output to Standard Output in the following format:
~~~
$s_0$
$\vdots$
$s_{T-1}$
~~~


<a href="https://img.atcoder.jp/ahc034/vImT4eac.html?lang=en&seed=0&output=sample">Show example</a>



Input Generation
--------
<details>
<p>
You are not required to understand this.
We recommend changing the seed value in the web visualizer to observe what kind of inputs are generated.
</p>
<p>
Let $\mathrm{rand}(L,U)$ be a function that generates a uniform random integer between $L$ and $U$, inclusive.
Let $\mathrm{noise}(y,x,\mathrm{seed})$ be a function that generates two-dimensional <a href="https://en.wikipedia.org/wiki/Perlin_noise">Perlin noise</a> scaled to the range between $-1$ and $1$ based on a random seed value $\mathrm{seed}$.
</p>
<p>
First, generate a seed value for Perlin noise generation as $\mathrm{seed}=\mathrm{rand}(0,2^{32}-1)$.  
</p>
<p>
Next, for each $(i,j)$, generate $h_{i,j}=\mathrm{round}(\mathrm{noise}(i/10,j/10,\mathrm{seed})\times 50)$.
If $h_{i,j}=0$ for all $(i,j)$, then regenerate $\mathrm{seed}$.
</p>
<p>
Let $S$ be the sum of $h_{i,j}$.
To make $S=0$, perform the following transformation.
</p>
<p>
Shuffle all coordinates $(0,0),(0,1),\cdots,(N-1,N-1)$ in a uniformly random order, and let the $k$-th coordinate be $(i_k,j_k)$.  
If $S>0$, then for each $k=0,1,\cdots,S-1$, decrease $h_{i_{k\% N^2},j_{k\% N^2}}$ by $1$.  
If $S<0$, then for each $k=0,1,\cdots,-S-1$, increase $h_{i_{k\% N^2},j_{k\% N^2}}$ by $1$.
</p>
</details>

Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc034/vImT4eac.html?lang=en">Web version</a>: This is more powerful than the local version providing animations.
- <a href="https://img.atcoder.jp/ahc034/vImT4eac.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc034/vImT4eac_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

Please be aware that sharing visualization results or discussing solutions/ideas during the contest is prohibited.

{sample example}
