TIME LIMIT: 1 minute
MEMORY LIMIT: 814 MB

Description
Create and print an 8 × 14 table (grid) whose cells contain only digits 0–9. We say a number can be “read” from the grid if it can be formed as follows:

- Start at any cell in the grid.
- Move to an adjacent cell each step, where adjacency includes up, down, left, right, and the four diagonals (8 directions in total).
- Append the digit in each visited cell, in order, to form the number.

Additional rules:
- You may revisit cells you have previously visited while forming a number.
- You may not skip over cells (i.e., you cannot move to a non-adjacent cell in one step).
- You may not remain in the same cell to append additional digits without moving (i.e., no “stay” moves).

Example (conceptual, 2 × 3 grid):
Starting at a cell containing 1, moving down, then right, then up-left, then left, you could form 12314, so the number 12,314 is readable on that grid. If you begin at the cell with 4 and traverse the reverse path, you could read 41,321. However, you cannot jump to skip cells (so 46 would be unreadable if it requires a jump), and you cannot repeatedly stay on one cell (so 11 would be unreadable if it requires staying).

Input
There is no input. DO NOT TRY TO READ INPUT. WE WILL GIVE YOU THE WRONG ANSWER VERDICT.

Output
Print an 8 × 14 grid consisting only of digits 0–9.

Scoring
If every integer from 1 up to X (inclusive) can be read from your printed grid, but X+1 cannot, then your score is X.

Sample Output
10203344536473
01020102010201
00000000008390
00000000000400
00000000000000
55600000000089
78900066000089
00000789000077

Explanation of Sample
For the grid above, all integers from 1 through 112 can be read, but 113 cannot, so the score is 112.

NOTE: THE CODE IS REQUIRED, NOT THE ACTUAL OUTPUT, AND YOU CODE HAS 1 MINUTE TO GENERATE AN 8X14 GRID. AND DO NOT JUST HAVE THE CODE PRINT OUT A GRID YOU PREDETERMINED.