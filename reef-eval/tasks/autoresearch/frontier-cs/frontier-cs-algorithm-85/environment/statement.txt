Maze
Time limit: 2s
Memory limit: 512MB

3.1 Background
The subject of this work is SCP-167. It is recommended to review information about SCP-167 before proceeding with the work.

3.2 Description
SCP-167 can be roughly described as a point connected to an infinitely deep binary tree, as shown in the figure below.

Level 0 o
|
Level 1 o
/
Level 2 o ...
/
Level 3 o ...
o
o
o

We define the point at Level 0 as the exit of SCP-167. To prevent researchers from getting lost inside SCP-167, you need to write a pathfinding system to help SCP-167 researchers find the exit.
For a researcher lost in SCP-167, because the three edges connected to their current node are highly similar, they cannot tell which edge leads towards the exit.
In SCP-167, one becomes lost as soon as they cannot determine the absolute position of their current node. At this point, all you know is the relative path between points.
To easily describe relative paths, we can assume each edge is one of three colors: red, yellow, or blue, and the three edges connected to any single node are all different colors.
Thus, each move can be represented by a color, indicating the edge of that color.
Simultaneously, you have a ranging device that can measure the distance between your current node and the exit.

Assume you are lost in SCP-167. You only know the initial distance initialDeep between your current node and the exit, but not the absolute position of this node. Please write a pathfinding system to locate the exit.

3.3 Task Description
To simulate being lost, we have prepared an interaction library. You will read initialDeep from the standard input, which represents the distance between your initial position and the exit. Here, initialDeep ≤ 10^4. You can perform the following two operations by outputing to the interactor by standard output. After each operation, you need to flush the output and read the returned value of the interactor from the standard input.

"move c"

where c is an integer 0, 1, or 2, which represent red, yellow and blue respectively. You can perform such operation for at most
10^5
  times. Once you have reached the exit, you must not move again. The interactor will return 1 if you reach the exit after the move. Otherwise, it will return 0.

"query"

The interactor will return the distance between your current node and the exit. The number of times you make a query will determine your score. Details are shown below.

3.6 Grading Method

The the number of queries exceed 10^5, you will get a score of zero. Otherwise, the less queries, the higher score you will get.

The time and memory limits given in the problem statement are the total resources available for your code and the interaction library combined.
We guarantee that for any legal data and calls within the limits, any version of the interaction library (including those distributed to contestants and those used for final evaluation) will not use more than 0.1 seconds of CPU time or more than 10MB of memory. This means contestants have at least 1.9 seconds of CPU time and at least 502MB of memory available for their code.