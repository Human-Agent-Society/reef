Story
--------
To celebrate the 10th anniversary of AtCoder Inc., we plan to hold an anniversary party with users invited.
At the party, CEO Takahashi will cut a giant cake with a knife in straight lines and distribute pieces to the attendees.
There are many strawberries on the cake, and he wants to distribute a piece containing $d$ strawberries to an attendee who has been participating in AtCoder's contests for $d$ years.
Please find a way to cut the cake which maximizes the number of pieces to be distributed under the specified upper limit on the number of cuts.
Note that Takahashi will eat all the pieces that are not distributed, so any leftovers are allowed.

<svg height="400" id="vis" viewBox="-5 -5 810 810" width="400" xmlns="http://www.w3.org/2000/svg">
<symbol viewBox="0 0 950.001 950.001" id="strawberry">
<path d="M97.96,525.152c15.452,203.891,186.558,424.849,377.056,424.849c190.543,0,361.649-220.958,377.011-424.849
	c9.989-131.284-46.628-220.44-140.062-271.815c35.595-30.553,66.702-66.721,80.655-83.699c4.372-5.32,2.354-10.907-4.396-12.268
	c-39.749-8.012-166.491-31.186-228.326-14.604c-31.713,8.513-48.912,20.782-57.246,35.299
	c-1.297-31.134-6.179-74.02-23.199-112.543c-16.24-36.645-28.512-53.579-36.388-61.404c-4.885-4.853-15.573-4.934-22.097-2.728
	l-69.683,23.568c-6.522,2.206-6.88,6.636-1.195,10.52c22.261,15.207,72.626,58.1,82.958,140.188
	c-8.871-13.48-25.892-24.86-55.902-32.9c-61.836-16.582-188.578,6.592-228.327,14.604c-6.75,1.36-8.762,6.944-4.384,12.259
	c14.601,17.728,47.952,56.4,85.525,87.739C141.318,309.46,88.329,397.566,97.96,525.152z M188.194,438.437
	c8.82-23.672,21.091-44.972,37.081-62.975c52.137-2.688,130.693-9.903,175.845-28.983c22.75,26.819,52.97,48.349,68.731,58.656
	c5.763,3.77,15.125,3.402,20.542-0.847c12.677-9.941,35.233-29.218,54.729-54.181c50.706,18.053,132.671,24.034,180.598,25.979
	c15.591,17.805,27.198,38.771,35.931,62.039C745.259,443.075,733,457.697,733,475.681v39.867c0,22.011,17.963,39.863,39.956,39.863
	c0.537,0,0.97-0.289,1.465-0.289c-1.972,18.677-5.266,37.468-10.19,56.145c-7.303-9.499-18.295-16.012-31.191-16.012
	c-21.993,0-40.038,17.847-40.038,39.863v39.862c0,18.187,12.556,32.9,29.219,37.671c-11.241,19.686-24.111,38.119-38.128,55.566
	c-7.348-8.04-18.285-13.531-30.871-13.531c-21.992,0-40.22,15.792-40.22,35.3v35.271c0,3.426,1.535,6.563,2.567,9.696
	c-42.557,31.198-90.434,50.214-140.599,50.214c-37.538,0-73.772-10.641-107.59-29.047c4.256-6.316,7.621-13.531,7.621-21.704
	v-39.863c0-22.016-17.964-39.867-40.002-39.867c-22.035,0-39.998,17.852-39.998,39.867v6.674
	C230,736.995,184.79,644.588,175.43,555.386c21.991-0.062,39.57-17.868,39.57-39.838v-39.867
	C215,458.28,203.733,443.881,188.194,438.437z"></path>
<path d="M284.021,714.824c22.038,0,39.979-17.828,39.979-39.845v-39.862c0-22.017-17.94-39.863-39.979-39.863
	c-21.994,0-40.021,17.847-40.021,39.863v39.862C244,696.996,262.026,714.824,284.021,714.824z"></path>
<path d="M341.998,555.41c22.039,0,40.002-17.852,40.002-39.863v-39.866c0-22.015-17.964-39.862-40.002-39.862
	c-22.034,0-39.998,17.847-39.998,39.862v39.867C302,537.559,319.962,555.41,341.998,555.41z"></path>
<path d="M435,714.824c21.993,0,40-17.828,40-39.845v-39.862c0-22.017-18.007-39.863-40-39.863c-22.039,0-40,17.847-40,39.863
	v39.862C395,696.996,412.96,714.824,435,714.824z"></path>
<path d="M502.002,555.41c21.993,0,39.998-17.852,39.998-39.863v-39.866c0-22.015-18.005-39.862-39.998-39.862
	c-22.039,0-40.002,17.847-40.002,39.862v39.867C462,537.559,479.963,555.41,502.002,555.41z"></path>
<path d="M542,674.979c0,22.017,17.986,39.845,39.979,39.845c21.993,0,40.021-17.828,40.021-39.845v-39.862
	c0-22.017-18.027-39.863-40.021-39.863c-21.993,0-39.979,17.848-39.979,39.864V674.979z"></path>
<path d="M653.525,555.41c21.992,0,39.475-17.852,39.475-39.863v-39.866c0-22.015-17.481-39.862-39.475-39.862
	c-21.994,0-39.525,17.847-39.525,39.862v39.867C614,537.559,631.532,555.41,653.525,555.41z"></path>
<path d="M487.002,744.335c-22.04,0-40.002,17.851-40.002,39.867v39.863c0,21.991,17.963,39.844,40.002,39.844
	c21.993,0,39.998-17.853,39.998-39.844v-39.863C527,762.187,508.995,744.335,487.002,744.335z"></path>
</symbol>
<rect fill="white" height="810" width="810" x="-5" y="-5"></rect>
<circle cx="400" cy="400" fill="none" r="400" stroke="black" stroke-width="2"></circle>
<g>
<title>
(-5000, 5000)
d = 1
</title>
<use xlink:href="#strawberry" x="150" ,="" y="150" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(2000, 6000)
d = 3
</title>
<use xlink:href="#strawberry" x="430" ,="" y="110" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(5000, 7000)
d = 3
</title>
<use xlink:href="#strawberry" x="550" ,="" y="70" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(7000, -1000)
d = 3
</title>
<use xlink:href="#strawberry" x="630" ,="" y="390" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(0, 0)
d = 1
</title>
<use xlink:href="#strawberry" x="350" ,="" y="350" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(-4000, -5000)
d = 2
</title>
<use xlink:href="#strawberry" x="190" ,="" y="550" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(1000, -8000)
d = 2
</title>
<use xlink:href="#strawberry" x="390" ,="" y="670" width="100" height="100" fill="red"></use>
</g>
<g>
<title>
(0, -2500) - (1, -2500)
</title>
<line stroke="black" stroke-width="1" x1="12.701665379258339" x2="787.2983346207417" y1="500" y2="500"></line>
</g>
<g>
<title>
(-3000, 0) - (-2999, 2)
</title>
<line stroke="black" stroke-width="1" x1="131.67472617169588" x2="476.3252738283041" y1="696.6505476566083" y2="7.349452343391749"></line>
</g>
<g>
<title>
(0, 5000) - (1, 4998)
</title>
<line stroke="black" stroke-width="1" x1="305.644042258373" x2="654.3559577416269" y1="11.288084516746094" y2="708.7119154832538"></line>
</g>
</svg>

In the above example, by cutting the cake with seven strawberries in three straight lines, we obtain two pieces with one strawberry, one piece with two strawberries, and one piece with three strawberries.

Problem Statement
--------
There is a circular cake with a radius of $10^4$ centered at the origin and $N$ strawberries on top of it.
The center of the $i$-th strawberry is at the coordinates $(x_i, y_i)$ and satisfies $x_i^2+y_i^2<10^8$.
Takahashi can cut the cake in at most $K$ straight lines (not segments), which may intersect each other.
You should specify the line to be cut as a straight line passing through two different integer coordinates $(p_x,p_y)$ and $(q_x,q_y)$ satisfying $-10^9\leq p_x,p_y,q_x,q_y\leq 10^9$.
The two specified points can be outside of the cake.
Because he is clumsy, he cannot stop or curve a single cut in the middle of the cut.

For each $d=1,2,\cdots,10$, you are given the number $a_d$ of attendees who have been participating in AtCoder's contests for $d$ years.
Let $b_d$ be the number of pieces with $d$ strawberries on them.
Then we can distribute $\sum_{d=1}^{10} \min(a_d,b_d)$ pieces to attendees.
Here, the $i$-th strawberry belongs to a piece if and only if its center $(x_i, y_i)$ is contained inside (excluding the circumference) of the piece.
If a strawberry is cut in a straight line that passes through its center, it belongs to no pieces.

Scoring
--------
Let $b_d$ be the number of pieces with $d$ strawberries on them.
Then, you will get the following score.

$\mathrm{round}\left(10^6 \frac{\sum_{d=1}^{10}\min(a_d,b_d)}{\sum_{d=1}^{10} a_d}\right)$

If the number of cuts exceeds $K$ or you specify an invalid line, it will be judged as <span class='label label-warning' data-toggle='tooltip' data-placement='top' title="Wrong Answer">WA</span> .

There are 100 test cases, and the score of a submission is the total score for each test case. If you get a result other than <span class='label label-success' data-toggle='tooltip' data-placement='top' title="Accepted">AC</span> for one or more test cases, the score of the submission will be zero. The highest score obtained during the contest time will determine the final ranking, and there will be no system test after the contest. If more than one participant gets the same score, the ranking will be determined by the submission time of the submission that received that score.


Input
--------
Input is given from Standard Input in the following format:

~~~
$N$ $K$
$a_1$ $a_2$ $\cdots$ $a_{10}$
$x_1$ $y_1$
$\vdots$
$x_N$ $y_N$
~~~

- The number of strawberries $N$ is equal to the sum of the attendees' AtCoder years. That is, $N=\sum_{d=1}^{10} d\times a_d$.
- For all test cases, the upper limit on the number of cuts is fixed to $K=100$.
- The number $a_d$ of attendees who have been participating in AtCoder's contests for $d$ years satisfies $1\leq a_d\leq 100$.

Output
--------
Let $k (\leq K)$ be the number of cuts and let $(p_x^i, p_y^i), (q_x^i, q_y^i)$ be the two points specifying the $i$-th line, then output to Standard Output in the following format.

~~~
$k$
$p_x^1$ $p_y^1$ $q_x^1$ $q_y^1$
$\vdots$
$p_x^k$ $p_y^k$ $q_x^k$ $q_y^k$
~~~


<a href="https://img.atcoder.jp/ahc012/f756367b32.html?lang=en&seed=0&output=30%0A-1812+-1984+-9663+5131%0A-2586+7859+7423+-4798%0A193+8457+-300+2680%0A-149+-9041+6425+84%0A3329+-5601+-2055+-5156%0A4701+3753+-2852+-2578%0A-1417+-9909+-5593+-5639%0A-7013+2309+-7622+6652%0A-5296+-5647+-9646+-1031%0A-1275+8332+6134+1727%0A-6996+2724+933+4067%0A1134+3022+-9377+2501%0A6487+-7016+-6730+4775%0A9141+-7928+7794+-4177%0A-5164+-3440+-2341+-6820%0A-5940+2165+-1255+6089%0A1691+-1616+347+-8205%0A-7966+-6024+-2883+5136%0A6899+3888+1191+6179%0A-762+7446+2182+2919%0A-6843+1720+-8636+-9995%0A-2997+-6378+4359+-2044%0A-7094+2392+2381+558%0A-3241+7718+-2179+-1519%0A-358+247+-547+-5282%0A-376+-1900+7187+3663%0A7470+1828+4310+-103%0A-2289+7865+8712+-1962%0A-6942+-5180+-1909+326%0A3266+3617+3585+1090%0A">Show example</a>

You may output multiple solutions. If you output more than one solution, only the last one is used for scoring. You can compare solutions by using the web visualizer.


Input Generation
--------
<details>

Let $\mathrm{rand}(L,U)$ be a function that generates a uniform random integer between $L$ and $U$, inclusive.

#### Generation of $a_1,\cdots,a_{10}$
For each $i$, we independently generate $a_i=\mathrm{rand}(1, 100)$.


#### Generation of $x_i, y_i$
We generate the coordinates $(x_i, y_i)$ of the $i$-th strawberry uniformly at random from the integer points satisfying $x_i^2+y_i^2<10^8$.
If the Euclidean distance between the new point and an existing point is less than or equal to $10$, we re-generate the point.

</details>

Tools (Input generator and visualizer)
--------
- <a href="https://img.atcoder.jp/ahc012/f756367b32.html?lang=en">Web version</a>: This is more powerful than the local version and can display animations.
- <a href="https://img.atcoder.jp/ahc012/f756367b32.zip">Local version</a>: You need a compilation environment of <a href="https://www.rust-lang.org/">Rust language</a>.
	- <a href="https://img.atcoder.jp/ahc012/f756367b32_windows.zip">Pre-compiled binary for Windows</a>: If you are not familiar with the Rust language environment, please use this instead.

{sample example}