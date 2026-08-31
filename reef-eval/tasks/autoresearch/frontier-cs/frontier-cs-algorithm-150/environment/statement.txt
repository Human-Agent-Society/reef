Story
--------
Human genetic information is recorded in DNA with a double helix structure and is represented by a very long string consisting of four characters, `A`, `G`, `C`, and `T`.
Recently, alien cells were found in a meteorite.
As a result of research, it was found that the genetic information of this alien is recorded in a <a href="https://en.wikipedia.org/wiki/Torus">torus</a>-shaped material and is represented as an $N\times N$ matrix consisting of eight characters, `A`, `B`, `C`, `D`, `E`, `F`, `G`, and `H`.
Existing devices have failed to read this matrix directly, but they have succeeded in reading many one-dimensional subsequences that are contiguous vertically or horizontally.
Please estimate the matrix based on this information.

Problem Statement
--------
We define that a one-dimensional sequence $b=(b_0, \ldots, b_{k-1})$ is a **subsequence** of a matrix $a=(a_{i,j})_{0\leq i,j\leq N-1}$ if and only if there exists $(i, j)$ satisfying at least one of the following two conditions:

- For all $p=0,\ldots,k-1$, $b_p=a_{i,(j+p)\bmod N}$ holds. (horizontal match)
- For all $p=0,\ldots,k-1$, $b_p=a_{(i+p)\bmod N,j}$ holds. (vertical match)

Note that if the index is greater than or equal to $N$, we take the remainder divided by $N$ (in other words, $a$ is connected at the left and right ends, and the top and bottom ends).

Given $M$ strings $s_1, \ldots, s_M$ consisting of eight characters, `A`, `B`, $\ldots$, `H`, your goal is to find an $N\times N$ matrix consisting of characters `A`, `B`, $\ldots$, `H`, or `.` which contains as many of the given strings as possible as subsequences.
Here, `.` indicates an empty.

Scoring
--------
Let $c$ ($\leq M$) be the number of $i$'s such that $s_i$ is a subsequence of the output matrix, and let $d$ ($\leq N^2$) be the number of `.` contained in the output.
Then, you will obtain the following score.

- If $c<M$, $\mathrm{round}(10^8\times \frac{c}{M})$.
- If $c=M$, $\mathrm{round}(10^8\times \frac{2 N^2}{2 N^2-d})$.

If the output is illegal (not an $N\times N$ matrix, or containing a character other than `A`, `B`, $\ldots$, `H`, `.`), it is judged as `WA`.
There are 100 test cases, and the score of a submission is the total score for each test case.
If you get a result other than `AC` for one or more test cases, the score of the submission will be zero. The highest score obtained during the contest time will determine the final ranking, and there will be no system test after the contest.
If more than one participant gets the same score, the ranking will be determined by the submission time of the submission that received that score.


Input
--------
Input is given from Standard Input in the following format:

~~~
$N$ $M$
$s_1$
$\vdots$
$s_M$
~~~

- $N$ is fixed to 20 throughout all the test cases.
- $M$ is an integer satisfying $400\leq M\leq 800$.
- Each $s_i$ is a string consisting of characters `A`, `B`, $\ldots$, `H`, and its length is at least 2 and at most 12.

Output
--------
Output $N$ lines, each containing a string of length exactly $N$ consisting of characters `A`, `B`, $\ldots$, `H`, or `.`.

Input Generation
--------
Let $\mathrm{rand}(L,U)$ be a function that generates a uniformly random integer between $L$ and $U$, inclusive.

We first generate an $N\times N$ matrix $a=(a_{i,j})_{0\leq i,j\leq N-1}$ by generating each element uniformly randomly from `A`, `B`, $\ldots$, `H`.
We then generate a parameter $L=\mathrm{rand}(4, 10)$, which decides the average length of the strings, and the number of strings $M=\mathrm{rand}(400, 800)$.
We generate $M$ strings $s_1,\ldots,s_M$ by repeating the following process $M$ times.

We generate two integers $i=\mathrm{rand}(0, N-1)$ and $j=\mathrm{rand}(0, N-1)$ representing a starting point, an integer $d=\mathrm{rand}(0,1)$ representing a direction, and an integer $k=\mathrm{rand}(L-2, L+2)$ representing the length of the string.

- If $d=0$, we choose a horizontal subsequence $(a_{i,j},a_{i,(j+1)\bmod N},\ldots,a_{i,(j+k-1)\bmod N})$.
- If $d=1$, we choose a vertical subsequence $(a_{i,j},a_{(i+1)\bmod N,j},\ldots,a_{(i+k-1)\bmod N,j})$.

Tools
--------
- <a href="https://img.atcoder.jp/ahc004/a45fa3f18ab177158bf5961b12872f93.zip">Inputs</a>: A set of 100 inputs (seed 0-99) for local testing, including the sample input (seed 0). These inputs are different from the actual test cases.
- <a href="https://img.atcoder.jp/ahc004/a42b6f0655821d8b384b31377108e5512.html">Visualizer on the web</a>
- <a href="https://img.atcoder.jp/ahc004/222362f13a30b1342bf79d0041bd4d39.zip">Input generator and visualizer</a>: If you want to use more inputs, or if you want to visualize your output locally, you can use this program. You need a compilation environment of <a href="https://www.rust-lang.org/ja">Rust language</a>.

{sample example}
