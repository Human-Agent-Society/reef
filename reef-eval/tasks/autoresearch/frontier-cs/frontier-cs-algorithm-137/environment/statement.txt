Title: Kangaroos

Goal
You must print a grid (“map”) on which a simple multi-agent movement game is played. The judge will run 500 random control sequences on your map. Your map is accepted if at least 125 of those sequences fail to gather all agents into a single cell.

World
- The world is an n-by-m grid (1 ≤ n, m ≤ 20).
- Each cell is either empty (1) or a wall (0).
- Initially, every empty cell contains exactly one kangaroo (agent).
- A move command is one of: U (up), D (down), L (left), R (right).
- On each command, all kangaroos move simultaneously:
  * If the adjacent cell in that direction exists and is empty, the kangaroo moves there.
  * Otherwise, it stays where it is.

Map requirements
- Grid size at most 20 × 20.
- There must be at least two empty cells.
- The subgraph of empty cells must be connected (you can go between any two empty cells by moving only through empty cells).
- The subgraph of empty cells must be acyclic (no cycles). In other words, the empty cells form a tree.

Judging
- The judge generates 500 random control strings, each of length 50,000.
- Each character in a control string is chosen independently and uniformly from {U, D, L, R}.
- In 1 string, if after applying all 50,000 moves there are still at least two kangaroos in different cells (i.e., not all agents have gathered into one cell), then you earn 1 point for that testcase.
- Your goal is to try and maximize your score over 500 random testcases.
- If your map violates any of the legality rules above (size, connectedness, no cycles, etc.), it is rejected regardless of performance.

Input
- There is no input. You only print the map.

Output
1) Print two integers: n m (1 ≤ n, m ≤ 20).
2) Then print n lines, each a string of length m consisting only of characters ‘0’ and ‘1’.
   - ‘1’ means the cell is empty.
   - ‘0’ means the cell is a wall.

Example (format only; not necessarily valid as a final map)
3 4
1111
1010
1100

Notes
- The judge’s random strings are produced by a standard uniform choice among U, D, L, R (e.g., typical rand()%4 mapping).
- Make sure your map satisfies all “Map requirements”; otherwise it will be rejected even if it performs well on the random tests.
- When printing out the grid, you should actually provide a c++ code which will print out the grid when run.
