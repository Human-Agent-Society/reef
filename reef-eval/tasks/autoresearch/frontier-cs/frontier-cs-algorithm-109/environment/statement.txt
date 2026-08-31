Time Limit: 1 second
Memory Limit: 128 MB

Description
A knight moves on an N×N chessboard and must try to visit every square exactly once (no revisiting). Given the board size and the starting square, output the longest possible path.
Note that this problem doesn't require a full complete path. It only asks you to find the longest path possible, and output that path.

Constraints
- 6 ≤ N ≤ 666

Input
- The first line contains the integer N.
- The second line contains two integers r c — the starting position (row and column), 1-indexed.

Output
- The first line of output should be an integer l: the length of the path you found(remember to include r0, c0)
- Then output l lines. Each line must contain two integers r c (1-indexed), the successive positions visited by the knight, starting with the given starting position.
When outputting the answer, make sure to not have an extra empty line at the end of the output after printing out the final move.
- The first pair of coordinates in your path must be the starting points r0 and c0.

Sample Input
6
1 1

Sample Output
36
1 1
3 2
5 1
6 3
5 5
3 6
2 4
1 6
3 5
5 6
4 4
5 2
3 1
1 2
3 3
2 1
1 3
2 5
4 6
6 5
5 3
6 1
4 2
5 4
6 6
4 5
6 4
4 3
6 2
4 1
2 2
1 4
2 6
3 4
1 5
2 3