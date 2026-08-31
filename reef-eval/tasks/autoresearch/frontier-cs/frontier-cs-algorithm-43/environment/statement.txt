Bigger Sokoban 40k
Time Limit: 2 seconds
Memory Limit: 1024 MB

Problem
Sokoban is a famous puzzle game where a player moves around in a grid and pushes boxes to their storage locations.

Bigger Sokoban is a variation of Sokoban where both the boxes and the storage locations are larger than 1×1.
In this version, both boxes and storage locations are 2×2 in size.

The rules are the same as in the original Sokoban:

- Each cell in the grid is either empty or a wall.
- Some 2×2 areas of empty squares contain a box, and some 2×2 areas are marked as storage locations.
- The player may move up, down, left, or right into adjacent empty squares, but cannot pass through walls, boxes, or go outside the grid.
- If the player moves into a box, the box is pushed one square in that direction, provided that doing so does not push it into a wall, another box, or outside the grid.
- Boxes cannot be pulled.
- The number of boxes equals the number of storage locations.
- The puzzle is solved when all boxes are placed on the storage locations.

The grid must satisfy the following constraints:
- The grid contains one box and one storage location (each 2×2).
- The player, the box, and the storage location must not overlap.
- 1 ≤ N, M, N + M ≤ 100

Input
There is no input for this problem.

Output
On the first line, print two integers N and M, the grid size.
Then print N lines of length M, describing the grid.
Each character must be one of the following:
- '.' : empty square
- '#' : wall
- 'P' : player
- 'B' : box (2×2 block)
- 'S' : storage location (2×2 block)

The grid must contain exactly:
- 1 player (P)
- 4 'B' cells forming a single 2×2 square
- 4 'S' cells forming a single 2×2 square
The grid must be solvable.

Example Output
(Note: This is just an example of correct format.)
5 6
....SS
....SS
.#BB#.
..BB.P
......

Scoring
Your program must output a valid and solvable grid according to the above format.
Your solution will be evaluated by a checker program that computes the minimum number of moves required to solve your grid.

Scoring rule:
- 63,000 moves → 100 points
- 0 moves → 0 points
- Linear scaling between them.

Formula:
score = min(100, max(0, moves / 630))

Task
Write a C++ program that outputs one valid grid satisfying the above conditions to standard output.
Your goal is to maximize the minimum number of moves needed to solve the puzzle.
