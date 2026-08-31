Pen

This is an I/O interactive problem. I/O interaction refers to interactive problems, where the program communicates with a special judge during execution instead of producing all output at once. In these problems, the program sends queries (output) to the judge and must immediately read responses (input) before continuing. The solution must strictly follow the input-output protocol defined in the problem statement, because any extra output, missing flush, or incorrect format can cause a wrong answer. Unlike standard problems, interactive problems require careful handling of I/O, synchronization, and flushing to ensure smooth communication between the contestant’s code and the judge.

There are n pens, numbered from 0 to n − 1.
Each pen contains a certain amount of ink — let the i-th pen have p_i units of ink.
You only know that (p_0, p_1, …, p_{n-1}) is a permutation of 0 … n − 1, but you do not know the exact values of p_i.

Your task is to choose two pens such that together they have at least n units of ink remaining.

You cannot directly query the amount of ink in any pen.
Instead, you can only “try to write” with a pen — then you will learn whether it can still write or not.
If it can write, it will consume 1 unit of ink.

Note that the requirement refers to the remaining ink in the two chosen pens:
when you finally select them, their total remaining ink must be at least n.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1001). The description of the test cases follows.

Each test case contains one integer n (10 ≤ n ≤ 25), the number of pens.

Interaction

To use a pen, output one line. First output 0 followed by a space, then output the number of the pen you want to use. After flushing your output, your program should read a single integer x, x = 0 means that the pen was already empty, x = 1 means the pen still had ink before consuming 1 unit of ink (can be empty now).

If you want to select two pens, output one line. First output 1 followed by a space. Then output the number of those two pens, separated by a space. Note that they cannot be the same pen. After flushing your output, you should start processing the next test case (or end the code if there is no next test case).