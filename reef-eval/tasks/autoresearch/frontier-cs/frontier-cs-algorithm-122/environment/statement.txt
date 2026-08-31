This is an interactive problem.

The RiOI Team has recently developed a text editor named RiOI Editor. The editor works with exactly one integer parameter W
 — the width of each line. It is known that 1≤W≤10^5
.

As you cannot understand the RiOI Language, from your point of view, words differ from each other only by their length. Hence, an article of length n
 is defined as a sequence a
 consisting of n
 positive integers, where ai
 is the length of the i
-th word in the article. The RiOI Editor displays the article [a1,a2,…,an]
 on screen as follows:

If max(a1,a2,…,an)>W
, the editor is unable to display the article;
Otherwise, the editor is able to display the article by the following process:
Initially, l=1
, and s=0
. During the whole process, l
 always denotes the current number of lines in the editor, and s
 always denotes the sum of lengths of words in the last line;
Then, for each 1≤i≤n
:
If s+ai≤W
, the word is inserted at the end of the current line. Thus, l
 remains unchanged, and s
 gets increased by ai
.
Otherwise, the word is inserted into a new line. Thus, l
 becomes l+1
, and s
 becomes ai
.
The number of lines needed to display the article is the final value of l
.
You are very interested in the editor, so you decide to find out the value of W
 by inputting some articles into the editor and observing the number of lines needed to display each article.

Formally, you can query the jury at most 2
 times. In each query, you input an article [a1,a2,…,an]
 (1≤n≤10^5
) to the editor, and the jury will respond to you with:

The number of lines needed to display the article, if the editor is able to display it;
0
, if the editor is unable to display the article.
Input
Each test contains multiple test cases. The first line contains the number of test cases t
 (1≤t≤10
). The description of the test cases follows.

Interaction
For each test case, you can make up to 2
 queries to find out the value of W
. It is guaranteed that 1≤W≤10^5
.

To make a query, you should print a new line in the following format:

? n a1 a2 … an
 (1≤n,ai≤10^5
) — the article you input to the editor.
At the end of each query, the jury will print an integer, as described in the statements.

You need to minimize the sum of length of the articles in your queries. That is, the sum of n in your queries. A smaller sum will result in a better score.

To report that you have found the value of W
, print a new line in the following format:

! W
 — the parameter of the editor.
Printing the answer does not count as one of the 2
 queries.

After that, proceed to process the next test case or terminate the program if it was the last test case.

After printing each query do not forget to output the end of line and flush∗
 the output. Otherwise, you will get Idleness limit exceeded verdict.

If, at any interaction step, you read −1
 instead of valid data, your solution must exit immediately. This means that your solution will receive Wrong answer because of an invalid query or any other mistake. Failing to exit can result in an arbitrary verdict because your solution will continue to read from a closed stream.