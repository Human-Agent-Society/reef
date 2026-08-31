Story
--------
AtCoder's CEO Takahashi prepares for Halloween tomorrow.
At AtCoder's Halloween party, Takahashi will dress up in disguise and receive a piece of candy from 100 employees in turn by saying, "trick or treat!"
He prepares a square box that can contain $10\times 10$ pieces of candy in a grid pattern, and each employee puts a piece of candy in an empty space so that no candies overlap.
There are $3$ types of candies: strawberry, watermelon, and pumpkin flavors.
He knows which flavor of candy each employee will put in by preliminary survey, but he doesn't know where each employee will put it.
Since he is a clean freak, he will move the pieces of candy by tilting the box forward, backward, left, or right, just once for each piece of candy received, and eventually wants to make sure that the same type of candy is clustered together as much as possible.
Please help him by writing a program to determine the direction to tilt.

<figure>
	<img src="./images/b639c75d_1.gif">
	<figcaption>Example for $5\times 5$</figcaption>
</figure>


Problem Statement
--------
There is a box that can contain $10\times 10$ pieces of candy in a grid pattern.
The box is initially empty, and $100$ pieces of candy will be placed in order.
There are $3$ flavors of candy, and we know in advance the flavor $f_t (1\leq f_t\leq 3)$ of the $t$-th candy.
On the other hand, we do not know in advance to which cell each candy will be placed, and it will be chosen uniformly at random among the empty cells.
You cannot change the order in which the pieces of candy are received.

Each time you receive one piece of candy, you must tilt the box forward, backward, left, or right exactly once.
When you tilt the box, each piece of candy moves in that direction simultaneously until it reaches the edge or hits another candy.
For example, if you tilt the box forward in the state shown in the left figure, the box will be in the state shown in the right figure.

![](./images/b639c75d_2.png)
![](./images/b639c75d_3.png)

Scoring
--------
We define the connectivity of pieces of candy as follows and consider the connected components.
> Two pieces of candy are connected if and only if they are of the same flavor and can reach each other through only pieces of candy of the same flavor in the four directions (front, back, left, right).

For example, the state in the figure below consists of $7$ connected components of size $\\{1, 1, 2, 2, 4, 6, 9\\}$.

![](./images/b639c75d_4.png)

Let $n_1,\cdots,n_k$ be the list of sizes of connected components in the final state after receiving 100 pieces of candy, and let $d_i$ be the total number of pieces of candy of flavor $i$.
Then the score for the test case is

\\[\mathrm{round}\left(10^6\times\frac{\sum_{i=1}^k n_i^2}{\sum_{i=1}^3 d_i^2}\right)\\]

Your task is to write a program to determine the tilting directions so that you can get as high a score as possible.

There are 200 test cases, and the score of a submission is the total score for each test case.
If your submission produces an illegal output or exceeds the time limit for some test cases, the submission itself will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> or <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span> , and the score of the submission will be zero.
The highest score obtained during the contest will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time (note that this is changed from previous short-term AHCs).


Input and Output
--------
First, the flavor of each piece of candy is given from Standard Input in the following format.
~~~
$f_1$ $f_2$ $\cdots$ $f_{100}$
~~~

Each $f_t$ is an integer value between $1$ and $3$, representing the flavor of the $t$-th piece of candy.

After reading the above information, repeat the following process $100$ times.

In the $t$-th process ($1\leq t\leq 100$), a single integer $p_t$ between $1$ and $101-t$ is given from Standard Input.
Let us number the empty cells from $1$ to $101-t$ in front-to-back and left-to-right priority, as shown in the example figure below.
Then the $t$-th piece of candy is placed in the $p_t$-th empty cell.

![](./images/b639c75d_5.png)

After reading $p_t$, by representing forward, backward, left, and right by `F`, `B`, `L`, and `R`, respectively, output a single character to Standard Output to indicate which direction to tilt the box.

<font color="red">**The output must be followed by a new line, and you have to flush Standard Output.**</font>
Otherwise, the submission might be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Time Limit Exceeded">TLE</span>.
<font color="red">**Note that $p_{t+1}$ will not be given until you output the $t$-th direction.**</font>
Since nothing happens at the 100th tilt, you may skip the output.

#### Example

<table class="table table-bordered">
<thead>
<tr>
<th>$t$</th>
<th>Input</th>
<th>Output</th>
</tr>
</thead>
<tbody>
<tr>
<td>Prior information</td>
<td><pre>2 2 1 3 1 2 1 2 1 $\cdots$ 3</pre></td>
<td></td>
</tr>
<tr>
<td>1</td>
<td><pre>3</pre></td>
<td><pre>R</pre></td>
</tr>
<tr>
<td>2</td>
<td><pre>98</pre></td>
<td><pre>B</pre></td>
</tr>
<tr>
<td>$\vdots$</td>
<td></td>
<td></td>
</tr>
<tr>
<td>100</td>
<td><pre>1</pre></td>
<td><pre>L</pre></td>
</tr>
</tbody>
</table>


<a href="https://img.atcoder.jp/ahc015/b639c75d.html?lang=en&seed=0&output=R%0D%0AB%0D%0AB%0D%0AR%0D%0AF%0D%0AR%0D%0AF%0D%0AR%0D%0AR%0D%0AF%0D%0AB%0D%0AL%0D%0AB%0D%0AL%0D%0AF%0D%0AF%0D%0AB%0D%0AF%0D%0AB%0D%0AL%0D%0AL%0D%0AL%0D%0AL%0D%0AB%0D%0AF%0D%0AF%0D%0AR%0D%0AR%0D%0AF%0D%0AL%0D%0AL%0D%0AB%0D%0AL%0D%0AL%0D%0AL%0D%0AB%0D%0AL%0D%0AR%0D%0AF%0D%0AL%0D%0AB%0D%0AL%0D%0AF%0D%0AF%0D%0AF%0D%0AL%0D%0AL%0D%0AR%0D%0AB%0D%0AB%0D%0AF%0D%0AL%0D%0AF%0D%0AR%0D%0AB%0D%0AL%0D%0AF%0D%0AF%0D%0AR%0D%0AL%0D%0AR%0D%0AL%0D%0AR%0D%0AR%0D%0AB%0D%0AR%0D%0AB%0D%0AR%0D%0AR%0D%0AF%0D%0AB%0D%0AF%0D%0AR%0D%0AR%0D%0AF%0D%0AB%0D%0AF%0D%0AB%0D%0AF%0D%0AR%0D%0AB%0D%0AF%0D%0AF%0D%0AF%0D%0AB%0D%0AB%0D%0AL%0D%0AL%0D%0AR%0D%0AF%0D%0AB%0D%0AL%0D%0AB%0D%0AF%0D%0AB%0D%0AR%0D%0AF%0D%0AF%0D%0AL%0D%0AL%0D%0A">Show example</a>


Input Generation
--------
Let $\mathrm{rand}(L,U)$ be a function that generates a uniform random integer between $L$ and $U$, inclusive.
Each $f_t$ is generated by $\mathrm{rand}(1,3)$.
Each $p_t$ is generated by $\mathrm{rand}(1,101-t)$.


Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc015/b639c75d.html?lang=en">Web version</a>: This is more powerful than the local version providing animations and manual play.
- <a href="https://img.atcoder.jp/ahc015/b639c75d.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc015/b639c75d_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

Please be aware that sharing visualization results or discussing solutions/ideas during the contest is prohibited.
