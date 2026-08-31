Guess Number

This is an interactive problem.

Vasya has thought of two secret integers a and b, both in the range [1, n]. Your task is to determine these two numbers by asking queries.

Each query consists of two integers x and y (both in the range [1, n]). The interactor will respond with one of the following:

- 0: Both x = a and y = b. You have successfully found the answer!
- 1: x is less than a (x < a)
- 2: y is less than b (y < b)  
- 3: x is greater than a OR y is greater than b (x > a or y > b)

Important: If multiple responses are true for your query, the interactor may choose any of them.

Your goal is to find both numbers a and b using as few queries as possible.

This problem is graded based on the number of queries you use. In order to receive any points, you must use no more than 10,000 queries. Your answer will be compared to a reference solution ref_queries. Your final score will be calculated as the average of 100 * min((ref_queries + 1) / (your_queries + 1), 1) across all test cases.

Input

There is only one test case in each test file.

The first line of the input contains an integer n (1 ≤ n ≤ 10^18) indicating the range of possible values.

Interaction

To ask a query, output one line containing two integers x and y (1 ≤ x, y ≤ n) separated by a space. After flushing your output, your program should read a single integer representing the interactor's response (0, 1, 2, or 3).

When you receive response 0, your program should terminate immediately.

To flush your output, you can use:
- fflush(stdout) (if you use printf) or cout.flush() (if you use cout) in C and C++.
- System.out.flush() in Java.
- stdout.flush() in Python.

Example

Input:
5

3

3

2

1

0
Output:

4 3

3 4

3 3

1 5

2 4

Time limit: 2 seconds
Memory Limit: 512 MB