# Palindrome Path

**Input file:** standard input  
**Output file:** standard output  
**Time limit:** 1 second  
**Memory limit:** 256 megabytes  

## Problem Description

Given an \( n \times m \) grid maze, each cell \((i,j)\) is either blank or blocked. George starts at cell \((sr,sc)\) and needs to visit all blank cells in the maze, finally ending at the exit cell \((er,ec)\). George can perform four types of moves, denoted as "L" (left), "R" (right), "U" (up), and "D" (down). When attempting a move, if the adjacent cell in that direction is blank and within bounds, George moves to it; otherwise, he stays in the current cell.

The move sequence must be a palindrome. That is, if the sequence is \( M_1, M_2, \cdots, M_k \) where each \( M_i \in \{L, R, U, D\} \), then for every \( i \) from 1 to \( k \), \( M_i = M_{k-i+1} \).

Your task is to find such a palindrome move sequence that visits all blank cells and ends at the exit cell. If multiple solutions exist, print any one. If no solution exists, output "-1". You need to minimize the number of moves.

Scoring:
Clamp(1.0 - (your sequence length - bound) / bound, 0, 1) where bound is a defined reachable solution number

## Input Format

The first line contains two integers \( n \) and \( m \) (\( 1 \leq n, m \leq 30 \)), the dimensions of the maze.

The next \( n \) lines each contain a binary string of length \( m \). A "1" indicates a blank cell, and a "0" indicates a blocked cell.

The following line contains four integers \( sr \), \( sc \), \( er \), \( ec \) (\( 1 \leq sr, er \leq n \), \( 1 \leq sc, ec \leq m \)), representing the start and exit cells. It is guaranteed that both cells are blank.

## Output Format

If a solution exists, print a string \( S \) (\( 0 \leq |S| \leq 10^6 \)) of moves on one line. If no moves are needed, output an empty line. If no solution exists, output "-1" on one line.

Do not include any extra spaces at the end of your output.

## Examples

### Example 1
**Input:**
```
2 2
11
11
1 1 2 2
```
**Output:**
```
RDLUULDR
```

**Explanation:**
The move sequence "RDLUULDR" results in the following path:
- Move right from (1,1) to (1,2)
- Move down from (1,2) to (2,2)
- Move left from (2,2) to (2,1)
- Move up from (2,1) to (1,1)
- Attempt up from (1,1) but stay (since i=1)
- Attempt left from (1,1) but stay (since j=1)
- Move down from (1,1) to (2,1)
- Move right from (2,1) to (2,2)

All blank cells are visited, and the path ends at (2,2).

### Example 2
**Input:**
```
2 2
10
01
1 1 2 2
```
**Output:**
```
-1
```

**Explanation:** George cannot reach (2,2) from (1,1) due to blocked cells.

## Constraints
- \( n, m \leq 30 \)
- The number of blank cells is at most \( 30 \times 30 = 900 \).
- The move sequence length must not exceed \( 10^6 \).

## Notes
- The start cell is considered visited at the beginning.
- If no moves are required (e.g., start and exit are the same, and only one blank cell), output an empty string.
- Ensure the output sequence is a palindrome and satisfies the visiting condition.