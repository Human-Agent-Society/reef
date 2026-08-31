“Da Bai” (Interactive)

There is an undirected simple graph with 100 vertices.

In each query, you can ask the interactor by choosing three distinct vertices a, b, c.  
The interactor will tell you how many edges exist among the three vertices {a, b, c}, i.e., the number of edges in the induced sub-graph on {a, b, c}. The answer will be an integer in {0, 1, 2, 3}.  
You must ensure that a, b, c are pairwise distinct.

Your goal: reconstruct the entire graph.

Implementation details

This is an interactive problem. As usual, you should submit a source code file that can compile.  
Initially, you should read no input.

You may issue queries by writing to standard output lines of the form  

? a b c

Then you must flush output, and read from standard input one integer in [0, 3], representing the interactor’s response.

When you have determined the graph, output a line  

!

Then output 100 lines, each with a length-100 binary string s_i (i from 1 to 100), describing the graph. Here s_{i,j} = ‘1’ if and only if there is an edge between vertex i and j in the graph.  
Remember to flush after your output.

Sample  
For convenience, in this sample we assume the graph has only **4** vertices. (In actual tests the graph has 100 vertices.)

Contestant prints     | Interactor replies  
---------------------|--------------------  
`? 1 2 3`             | 0  
`? 1 2 4`             | 2  
`? 1 3 4`             | 2  
`!`                  |  
0001  
0001  
0001  
1110  

Constraints  
- All test instances have exactly 100 vertices.  
- Let Q = the number of queries you make.  
  - If your program exceeds time limit, memory limit, or returns incorrect answer → score=0.
  - Otherwise, your score depends on Q:
    - score(Q) = 3400 / (Q + 1)
    - In other words, a solution with Q < 3400 is awarded the full score.
