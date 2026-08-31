Problem: Treasure Hunt

Time limit: 5 seconds

Memory limit: 256 MB

This is an interactive problem.

Imagine you are a treasure hunter exploring an ancient map represented as an undirected graph with n vertices (junctions) and m edges (roads). You start at a known fixed vertex. Your goal is to visit every vertex in the graph at least once to collect treasures.

However, a wicked wizard is working against you. While the structure of the graph remains constant, the wizard shuffles the order of the roads at every junction each time you arrive. You cannot see the labels of the adjacent vertices. Instead, when standing at a junction, you can only see:
- The degree of the current vertex.
- For each adjacent vertex: its degree and whether you have visited it before (indicated by a flag).

You must navigate the graph and collect all treasures as fast as possible. You are provided with a base_move_count, and your score depends on how close your move count is to this baseline.

Scoring:
Let moves be the number of moves you take.
If moves <= base_move_count, you get:
   100 - c * (sol_fraction - 1)
   where c = 90 / sqrt(base_fraction - 1)
   base_fraction = (base_move_count + 1) / n
   sol_fraction = (moves + 1) / n

If base_move_count < moves <= 2 * base_move_count, you get:
   20 * (1.0 - (moves + 1) / (base_move_count + 1)) points.

If moves > 2 * base_move_count, you get 0 points.

Interaction Protocol

First, the interactor prints an integer t (1 <= t <= 5), the number of maps to solve.
Then for each map, the interactor prints the graph description:
- Four integers: n, m, start, base_move_count.
  (2 <= n <= 300, 1 <= m <= min(n(n-1)/2, 25n), 1 <= start <= n).
- m lines describing the edges u, v (1 <= u, v <= n).

After the graph description, the interaction begins. The interactor prints vertex descriptions in the format:
   d deg_1 flag_1 deg_2 flag_2 ... deg_d flag_d
   where d is the degree of the current vertex, deg_i is the degree of the i-th neighbor, and flag_i (0 or 1) indicates if that neighbor has been visited.

To make a move, you must output a single integer i (1 <= i <= d), choosing the i-th neighbor described in the current line.
Remember to use the flush operation after each output.

The interaction ends for a map when:
- You have visited all vertices: The interactor prints "AC".
- You have exceeded the move limit: The interactor prints "F".
You must then proceed to the next map or terminate if it was the last one.

Example Input:
1
3 3 1 1000
1 2
2 3
3 1
2 2 0 2 0

2 2 0 2 1

2 2 0 2 1

AC

Example Output:





1

2

1