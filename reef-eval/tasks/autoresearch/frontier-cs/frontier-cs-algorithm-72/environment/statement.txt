Puzzle

There is a puzzle game. The board is a 6 × 6 grid with grooves in the tiles to allow vehicles to slide. Cars and trucks are both one square wide, but cars are two squares long and trucks are three squares long. Vehicles can only be moved forward or backward along a straight line on the grid. The goal of the game is to get the only red car totally out through the exit of the board by moving the other vehicles out of its way.
We give each vehicle of a puzzle a unique id, numbered from 1 to the number of vehicles, in which the red car's id is 1. The board information of a puzzle is represented by a 6 × 6 matrix, named board matrix. Each entry of a board matrix is the id of the vehicle placed on that groove, and the entries are filled with 0 if there exists no vehicle on those grooves. The exit of the board is located at the right end side of the 3rd row.
Moving a piece by one unit is called a step.
Given an initial puzzle, you can conduct a sequence of moves to form a new puzzle.
Your task is to maximum the minimum number of steps required to solve the new puzzle and output the sequence of moves to form the new puzzle.

Input
The input contains 6 lines, each line indicates the content (6 integers separated by a blank) of each row of a board matrix.

Output
First, output two integers in one line, representing the minimum number of steps required to solve the new puzzle and the number of steps to form the new puzzle.
Then, output the sequence of moves to form the new puzzle. Each move should be on a separate line in the format:
    vehicle_id direction 
where:
- vehicle_id is the id of the vehicle to move (1 to n)
- direction is one of: U (up), D (down), L (left), R (right)

Scoring
This problem uses a scoring system based on your solution's efficiency.
Let your_steps be the minimum number of steps required to solve the new puzzle in your solution, and ref_steps be the minimum number of steps required to solve the new puzzle in the reference solution.
Your score for each test case is calculated as follows:
   score = 100 * min((your_steps + 1) / (ref_steps + 1), 1)
Your final score is the average of your scores across all test cases.

Technical Specification
- There are at most 10 vehicles in the puzzle.
- The red car is always horizontal and positioned on the 3rd row.
- All vehicles can only move forward or backward along their orientation (horizontal vehicles move left/right, vertical vehicles move up/down).
- It is guaranteed there is a solution for each initial puzzle.

Example

Sample input:
0 0 0 0 0 0
0 0 0 0 0 0
0 1 1 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0

Sample output:
6 1
1 L

Time limit: 2 seconds
Memory limit: 1024 MB