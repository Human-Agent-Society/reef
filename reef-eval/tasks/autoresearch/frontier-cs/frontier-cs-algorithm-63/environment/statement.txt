Space Thief

This is an I/O interactive problem. I/O interaction refers to interactive problems, where the program communicates with a special judge during execution instead of producing all output at once. In these problems, the program sends queries (output) to the judge and must immediately read responses (input) before continuing. The solution must strictly follow the input-output protocol defined in the problem statement, because any extra output, missing flush, or incorrect format can cause a wrong answer. Unlike standard problems, interactive problems require careful handling of I/O, synchronization, and flushing to ensure smooth communication between the contestant’s code and the judge.

You are active as a thief in the JOI galaxy.

There are N stars, numbered from 0 to N − 1, in the JOI galaxy.
There are M warp devices, numbered from 0 to M − 1.
Each warp device i (0 ≤ i ≤ M − 1) connects two stars U_i and V_i bidirectionally.
It is possible to travel between any two stars using warp devices.

A key is hidden in one star, and a treasure box is hidden in another.
Your mission is to determine the numbers of the stars where the key and the treasure box are hidden.

To do this, you may ask up to 600 questions of the following form:

For each warp device i (0 ≤ i ≤ M − 1), choose one of two directions:

Allow travel only from U_i to V_i.

Allow travel only from V_i to U_i.

Then ask whether it is possible to travel from the star containing the key to the star containing the treasure box under these directed conditions.

Your goal is to identify the star A containing the key and the star B containing the treasure box while minimizing the number of questions. Let q be the number of queries you asked, your score will be (600 - q) / 600.

Constraints

2 ≤ N ≤ 10,000

1 ≤ M ≤ 15,000

0 ≤ A, B ≤ N − 1 and A ≠ B

0 ≤ Ui < Vi ≤ N − 1

All pairs (Ui, Vi) are unique

The graph is connected (travel possible between any two stars)

Input

The first line contains two integers: N and M.

For the next M lines, the i-th line contains 2 integers: U_{i-1} and V_{i-1}.

Interaction

To ask a query, output one line. First output 0 followed by a space, then output a sequence of m integers of 0 or 1 separated by a space. After flushing your output, your program should read a single integer x, x = 0 means that it is not possible to travel from the star A to the star B by using warp devices, x = 1 means that it is possible to travel from the star A to the star B by using warp devices.

If you want to guess the star A and the star B, output one line. First output 1 followed by a space. Then output A followed by a space, then output B. After flushing your output, your program should exit immediately.

Note that the answer for each test case is pre-determined. That is, the interactor is not adaptive. Also note that your guess does not count as a query.

To flush your output, you can use:

    fflush(stdout) (if you use printf) or cout.flush() (if you use cout) in C and C++.

    System.out.flush() in Java.

    stdout.flush() in Python.

Time limit: 2 seconds

Memory Limit: 1024 MB