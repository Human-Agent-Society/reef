# Number Loop Construction

You are given a fixed 12 by 12 clue template. Some cells are clue cells, and the
remaining cells are blank. Your task is to fill every clue cell with a digit and
construct a Number Loop puzzle.

The goal has two levels:

1. Make the puzzle have as few valid loop solutions as possible.
2. Among puzzles with the same number of valid loop solutions, maximize the
   number of clue cells equal to `1`.

## Number Loop Rules

A loop is drawn on the grid lines around the 12 by 12 cells.

* Each grid vertex must have degree either `0` or `2`.
* The selected grid edges must form exactly one non-empty closed loop.
* For every clue cell containing digit `d`, exactly `d` of its four surrounding
  grid edges must be selected.
* Blank cells impose no clue constraint.

A valid loop solution is one loop satisfying all clue constraints in your
output puzzle.

## Template

Your output must preserve the following blank positions exactly. Every `?`
position must be replaced by a digit.

```text
?   ?   ??? 
?? ??  ?   ?
? ? ?  ?   ?
? ? ?  ???? 
? ? ?  ?    
?   ?  ?    
            
?  ?   ?????
? ?      ?  
??   ? ? ?  
? ?  ? ? ?  
?  ? ??? ?  
```

There are 56 clue cells.

## Input Format

The input contains a single integer:

* `0`: each clue cell may be one of `0`, `1`, `2`, or `3`.
* `1`: each clue cell may be one of `1`, `2`, or `3`.

## Output Format

Output exactly 12 lines, each with exactly 12 characters.

* A blank position in the template must be output as a space.
* A `?` position in the template must be output as an allowed digit.

Trailing spaces are significant for the 12 by 12 format.

## Scoring

For each test case, the judge counts the number of valid loop solutions in your
constructed puzzle, up to a cap of six solutions. It also counts

```text
ones = number of clue cells equal to '1'
```

Let `u = ones / 56`.

The per-case score is:

```text
0 valid solutions: 0
6 or more valid solutions: 1 + 9 * u
5 valid solutions:         10 + 10 * u
4 valid solutions:         20 + 10 * u
3 valid solutions:         30 + 10 * u
2 valid solutions:         40 + 10 * u
1 valid solution:          50 + 50 * u
```

The final score is the average over the two test cases. A unique-solution puzzle
is therefore always better than any non-unique puzzle, and within the same
solution-count bucket, more `1` clues are better.

## Sample Output

The following output only illustrates the required 12 by 12 format. It is not
necessarily a high-scoring construction.

```text
0   0   000 
00 00  0   0
0 0 0  0   0
0 0 0  0000 
0 0 0  0    
0   0  0    
            
0  0   00000
0 0      0  
00   0 0 0  
0 0  0 0 0  
0  0 000 0  
```
