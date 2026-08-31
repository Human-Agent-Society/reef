Inverse Counting Path

Walk Alone is an expert in dynamic programming, but he gets bored with traditional dynamic programming problem like counting paths on a 2-dimension grid, so he wants to do it in reverse. The problem he raised is as follows:

On a 2-dimension grid of size n*n, originally you are on grid (1,1). The grid consists of 0 and 1, and you can only walk on the grid with number 1 in it. You can only go down or right, i.e. you can only increase your x or y by one. Also you cannot walk outside the grid.

Given the number x of ways to walk from (1,1) to (n,n), you need to construct a grid of n*n so that the ways to walk is exactly x. However, since Walk Alone's brain is too small to memorize such a big grid, you need to guarantee that the size of the grid n is equal to or smaller than 300. Specifically, your score will be (300 - n) / 300.

Input

The only line of the input contains one integer x (1<=x<=10^18), denoting the ways to walk.

Output

The first line of the output contains the size of the grid n. Remind that you need to guarantee 1<=n<=300.

The following n lines each contains n integers a_{i,j}∈{0,1} denoting the grid, where 0 denotes you cannot walk on the grid while 1 is on the contrast.

Example Input 1:

3

Example Output 1:

3
1 1 0
1 1 0
1 1 1

Example Input 2:

10

Example Output 2:

4
1 1 1 0
1 1 1 1
1 0 1 1
1 1 1 1