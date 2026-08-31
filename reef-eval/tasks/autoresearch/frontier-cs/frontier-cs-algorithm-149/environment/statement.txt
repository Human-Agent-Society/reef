Story
--------
AtCoder is developing a route navigation application that utilizes shortest path algorithms.
The service area is represented as a road network of 30x30 vertices connected in a grid.
When a user specifies the vertex of the current location and the vertex of the destination, the app will output the shortest path between them.
The trouble is that, even though the scheduled release date is approaching, the measurement of the length of each edge, which is essential for shortest path computations, is not finished at all.
Therefore, AtCoder decided to give up measuring the edge length in advance and allows the app to output paths that are not the shortest.
It should be possible to gradually improve the performance by estimating the length of each edge based on the information about the actual time users take to arrive at their destinations.

Problem Statement
--------
There is an undirected grid graph with 30x30 vertices with unknown edge lengths.
Let $(0, 0)$ denote the top-left vertex, and $(i, j)$ denote the vertex at the $i$-th row from the top and $j$-th column from the left.
Your task is to process the following query 1000 times.

In the $k$-th query, your program first receives the vertices $s_k=(si_k,sj_k)$ and $t_k=(ti_k,tj_k)$ from Standard Input in the following format:

~~~
$si_k$ $sj_k$ $ti_k$ $tj_k$
~~~

Then, your program should compute a path $P_k$ from $s_k$ to $t_k$.
Let `U`, `D`, `L`, and `R` represent the movement from $(i,j)$ to $(i-1,j)$, $(i+1,j)$, $(i,j-1)$, and $(i,j+1)$, respectively.
Output a string representing the path $P_k$ to Standard Output in one line.
**After the output, you have to flush Standard Output.** Otherwise, the submission might be judged as TLE.

After your program outputs a path, the judge program calculates the length $b_k$ of the path, generates a uniform random number $e_k$ between $0.9$ and $1.1$, and gives an integer value $\mathrm{round}(b_k\times e_k)$ to Standard Input.
By reading that integer, the $k$-th query completes, and you should proceed to the $k+1$-th query.


Examples
-----------------

<table class="table table-bordered">
<thead>
<tr>
<th align="left">Input</th>
<th align="left">Output</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><pre>3 19 16 17</pre></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"></td>
<td align="left"><pre>DDDDDDDDDDDDDLL</pre></td>
</tr>
<tr>
<td align="left"><pre>99561</pre></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><pre>26 18 13 18</pre></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"></td>
<td align="left"><pre>UUUUUUUUUUUUU</pre></td>
</tr>
<tr>
<td align="left"><pre>72947</pre></td>
<td align="left"></td>
</tr>
</tbody>
</table>


Scoring
--------
Let $a_k$ and $b_k$ be the lengths of the shortest path and the output path for the $k$-th query ($1\leq k\leq 1000$), respectively.
Then the score for the test case is

$\mathrm{round}(2312311\times \sum_{k=1}^{1000}0.998^{1000-k} \frac{a_k}{b_k})$

The score of a submission is the total score for each test case.
If your program outputs an illegal path (visiting the same vertex multiple times, going outside of 30x30, or not a path from $s$ to $t$), it is judged as `WA`.
After the contest is over, the final ranking will be determined by system tests against the last submission.

- Provisional tests consist of 100 test cases. If you get a result other than `AC` for one or more test cases, the score of the submission will be zero.
- System tests consist of 3000 test cases. If you get a result other than `AC` for some test cases, only the score for those test cases will be zero. We will publish seeds.txt (md5=0cf5051d586e7f62c0b3527f6f7fbb1c) after the contest is over.



Input Generation
--------
Let $\mathrm{rand}(L,U)$ be a function that generates a uniformly random integer between $L$ and $U$, inclusive.
We first generate two parameters $D=\mathrm{rand}(100, 2000)$ and $M=\mathrm{rand}(1, 2)$.
Let $h_{i,j}$ be the length of the edge between $(i, j)$ and $(i,j+1)$, and let $v_{i,j}$ be the length of the edge between $(i, j)$ and $(i+1,j)$.

#### Generation of $h_{i,j}$
1. For each $i\in\\{0,\ldots,29\\}$ and $p\in\\{0,\ldots,M-1\\}$, we independently generate a random integer $H_{i,p}=\mathrm{rand}(1000+D,9000-D)$.
2. For each $i\in\\{0,\ldots,29\\}$ and $j\in\\{0,\ldots,28\\}$, we independently generate a random integer $\delta_{i,j}=\mathrm{rand}(-D,D)$.
3. If $M=1$, for each $i\in\\{0,\ldots,29\\}$ and $j\in\\{0,\ldots,28\\}$, we set $h_{i,j}=H_{i,0}+\delta_{i,j}$.
4. If $M=2$, for each $i\in\\{0,\ldots,29\\}$, we generate a random integer $x_i=\mathrm{rand}(1,28)$, and then for each $j\in\\{0,\ldots,x_i-1\\}$, we set $h_{i,j}=H_{i,0}+\delta_{i,j}$, and for each $j\in\\{x_i,\ldots,28\\}$, we set $h_{i,j}=H_{i,1}+\delta_{i,j}$.

#### Generation of $v_{i,j}$
1. For each $j\in\\{0,\ldots,29\\}$ and $p\in\\{0,\ldots,M-1\\}$, we independently generate a random integer $V_{j,p}=\mathrm{rand}(1000+D,9000-D)$.
2. For each $i\in\\{0,\ldots,28\\}$ and $j\in\\{0,\ldots,29\\}$, we independently generate a random integer $\gamma_{i,j}=\mathrm{rand}(-D,D)$.
3. If $M=1$, for each $j\in\\{0,\ldots,29\\}$ and $i\in\\{0,\ldots,28\\}$, we set $v_{i,j}=V_{j,0}+\gamma_{i,j}$.
4. If $M=2$, for each $j\in\\{0,\ldots,29\\}$, we generate a random integer $y_j=\mathrm{rand}(1,28)$, and then for each $i\in\\{0,\ldots,y_j-1\\}$, we set $v_{i,j}=V_{j,0}+\gamma_{i,j}$, and for each $i\in\\{y_j,\ldots,28\\}$, we set $v_{i,j}=V_{j,1}+\gamma_{i,j}$.

#### Generation of $s_k$, $t_k$
The vertices $s_k$ and $t_k$ given in the query are chosen uniformly at random among all the vertices.
If the Manhattan distance between $s_k$ and $t_k$ ($|si_k-ti_k|+|sj_k-tj_k|$) is strictly less than 10, we repeat the random selection until the distance becomes at least 10.



Tools
--------
- <a href="https://img.atcoder.jp/ahc003/c1ae4a8996958aa31f5f9d3aa3f51033.zip">Local tester</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
- <a href="https://img.atcoder.jp/ahc003/e7eb814463364c249c93216eee64275.html">Visualizer</a>
- <a href="https://img.atcoder.jp/ahc003/499df4d8fb8c9326c7b718917d14f17a.zip">Inputs</a>: If you don't use the above local tester, you can instead use these 100 inputs (seed 0-99) for local testing. These inputs are different from the actual test cases. The inputs are in the following format, and you can use them by writing a judge program by yourself.

~~~
$h_{0,0}$ $\ldots$ $h_{0,28}$
$\vdots$
$h_{29,0}$ $\ldots$ $h_{29,28}$
$v_{0,0}$ $\ldots$ $v_{0,29}$
$\vdots$
$v_{28,0}$ $\ldots$ $v_{28,29}$
$si_1$ $sj_1$ $ti_1$ $tj_1$ $a_1$ $e_1$
$\vdots$
$si_{1000}$ $sj_{1000}$ $ti_{1000}$ $tj_{1000}$ $a_{1000}$ $e_{1000}$
~~~

#### Example of judge program (pseudo code)
~~~
string query(s, t, prev_result) {
	// WRITE YOUR SOLUTION HERE
}

int main() {
	if (LOCAL_TEST) {
		read_h_v();
	}
	prev_result = 0;
	score = 0.0;
	for (int k = 0; k < 1000; k++) {
		if (LOCAL_TEST) {
			read_s_t_a_e();
		} else {
			read_s_t();
		}
		path = query(s, t, prev_result);
		print(path);
		if (LOCAL_TEST) {
			b = compute_path_length(path);
			score = score * 0.998 + a / b;
			prev_result = round(b * e);
		} else {
			prev_result = read_result();
		}
	}
	if (LOCAL_TEST) {
		print(round(2312311 * score));
	}
	return 0;
}
~~~
