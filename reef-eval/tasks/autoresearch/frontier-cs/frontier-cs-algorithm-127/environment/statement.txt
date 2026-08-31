Time Limit: 1 second
Memory Limit: 256 mb

Interactive Problem

YOU MUST USE INTERACTIVE FORMAT FOR THIS PROBLEM.

You are standing in front of a row of n boxes, labeled 0 through n−1 from left to right.
Each box contains a prize that cannot be seen until the box is opened. There are v ≥ 2
different prize types, numbered from 1 to v in decreasing order of value:
type 1 is the most expensive (the diamond), and type v is the cheapest (a lollipop).
There is exactly one diamond in the boxes.

The number of cheaper prizes is much larger than the number of more expensive ones.
More precisely, for all 2 ≤ t ≤ v: if there are k prizes of type t−1, then there are
strictly more than k^2 prizes of type t.

Your task is to find the index i of the box containing the diamond using as few
queries as possible.

-------------------------------------------------------------------------------
Input
-------------------------------------------------------------------------------
Standard input (stdin) initially contains a single integer:
  n  — the number of boxes.

Box indices are 0-based: 0, 1, …, n−1.

-------------------------------------------------------------------------------
Interaction protocol (stdin/stdout)
-------------------------------------------------------------------------------
Your program communicates using standard input and standard output only.
There is no provided procedure to call; write a normal main program.

To ask about a box i (0 ≤ i < n):
  • Print a line:   ? i
  • Flush the output stream immediately.

Then read from stdin two integers a0 and a1 (on the same line):
  • a0  — among the boxes strictly to the left of i, the number of boxes that
           contain a more expensive prize than the one in box i.
  • a1  — among the boxes strictly to the right of i, the number of boxes that
           contain a more expensive prize than the one in box i.

When you have determined the index i of the diamond (type 1), finish by:
  • Printing a line:   ! i
  • Flushing the output, then terminating the program.

Notes:
  • You may output extra whitespace on lines you print, but each command must be
    on its own line.
  • After each query line you print ("? i"), you must read exactly two integers.
  • In some test cases, the judge may be adaptive: its answers may depend on the
    questions you ask, but it will always remain consistent with the constraints.

-------------------------------------------------------------------------------
Constraints
-------------------------------------------------------------------------------
3 ≤ n ≤ 200000.
Each prize type is an integer between 1 and v, inclusive.
There is exactly one prize of type 1 (the diamond).
For all 2 ≤ t ≤ v, if there are k prizes of type t−1, then there are strictly
more than k^2 prizes of type t.

-------------------------------------------------------------------------------
Subtasks
-------------------------------------------------------------------------------
Single subtask: n ≤ 200000. (All constraints above apply.)

-------------------------------------------------------------------------------
Example (verbal transcript; no image)
-------------------------------------------------------------------------------
Suppose n = 8 and the hidden prize types are:
  index:   0 1 2 3 4 5 6 7
  types:   3 2 3 1 3 3 2 3
Here, index 3 holds the diamond (type 1).

Possible interaction:
  (read) 8
  (you)  ? 0        → (judge replies) 0 3
          Meaning: left of 0 there are 0 more-expensive prizes; right of 0 there
                   are 3 more-expensive prizes than the prize at 0.

  (you)  ? 2        → (judge replies) 1 2
          Meaning: among indices {0,1}, exactly one is more expensive than 2’s;
                   among {3,4,5,6,7}, exactly two are more expensive than 2’s.

  (you)  ? 3        → (judge replies) 0 0
          Meaning: no box on either side is more expensive than 3’s prize.

  (you)  ! 3        (finish)

The exact sequence of queries you make can differ; this is only an illustrative
transcript of the format.

-------------------------------------------------------------------------------
Scoring
-------------------------------------------------------------------------------
Let q be the number of queries you print (i.e., the number of lines of the form "? i").
Your raw score for a test is:
  5000 − q   (clamped below at 0 if needed).
Your overall score is your average raw score divided by the best score.

-------------------------------------------------------------------------------
Output
-------------------------------------------------------------------------------
Print exactly one final line of the form:
  ! i
where i is the index (0 ≤ i < n) of the box containing the diamond.

IMPORTANT NOTE: Make sure to print out answer at the end of your code. Even if you did not find the diamond, print any arbitrary index before exiting.
