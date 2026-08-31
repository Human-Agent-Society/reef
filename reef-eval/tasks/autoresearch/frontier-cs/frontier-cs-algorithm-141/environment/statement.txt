Bakery Survey

This is an interactive problem.

Your city has n bakeries (where n is a power of 2), and bakery i specializes in one type of cake a_i.

You want to determine d — the number of distinct cake types available in the city.

You don't know the values of a_1, ..., a_n. However, your friend can help you by tasting cakes. Your friend has a memory capacity of k (where k is also a power of 2), which works as follows:

Your friend's memory is a queue S. You can perform two types of operations:

1. Query operation: Ask your friend to taste the cake from bakery c. This will:
   - Tell you whether a_c is already in S (the last k cake types tasted)
   - Add a_c to the end of S
   - If |S| > k, remove the front element from S

2. Reset operation: Clear your friend's memory, making S empty.

Your goal is to find d while minimizing the total cost of operations.

This problem is graded based on the total cost of operations. The cost is calculated as:
Total Cost = (number of resets) × n + (number of queries) + 1

Your answer will be compared to a reference solution ref_cost. Your final score will be calculated as the average of 100 × min(ref_cost / your_cost, 1) across all test cases.

You must use at most 100,000 operations in total.

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 1024, both k and n are powers of 2).

Interaction

To perform an operation, output one line in one of the following formats:

? c — Ask your friend to taste the cake from bakery c (1 ≤ c ≤ n).
R — Reset your friend's memory.

After a query operation, read a single character:
- Y (Yes) if a_c is in the memory S
- N (No) if a_c is not in the memory S

When you have found the answer, output:
! d — where d is the number of distinct cake types.

After printing the answer, your program should terminate immediately.

To flush your output, use:
- fflush(stdout) or cout.flush() in C++
- System.out.flush() in Java
- stdout.flush() in Python

Example 

Input:
4 2
N
N
Y
N
N
N
N

Output:
? 1
? 2
? 3
? 4
R
? 4
? 1
? 2
! 3

Time limit: 4 seconds
Memory limit: 512 MB