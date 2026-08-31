Time Limit: 2 s
Memory Limit: 1024 MB
This is an interactive problem.
Bitaro found out that a string S of length N was written on the stone slate. Each character of the string is either ‘0’ or ‘1’. However, he does not yet know each character of the string S.
Bibako found out how to use the machine. To use it, we put the stone slate on the machine, input an integer m and two sequences of integers a, b, and make a query. Here the integer m and the sequences of integers a, b should satisfy:
- 1 <= m <= 1002
- Both of the lengths of the sequences a, b are equal to m.
- Every element of the sequences a, b is an integer between 0 and m - 1, inclusive.
If we put the stone slate on the machine, input an integer m and two sequences of integers a, b, and make a query, the machine will operate as follows and will show an integer.
 1. The machine sets 0 in the memory area of the machine.
 2. The machine performs the following N operations. The (i + 1)-th (0 <= i <= N - 1) operation proceeds as follows.
Let x be the current integer set in the memory area of the machine. The machine reads the character S_i (0<=i<N).
 - If S_i is ‘0’, the machine sets a_x in the memory area of the machine. Here, a_x is the x-th element of the sequence a, if we count the elements of the sequence a so that the first element is the 0-th element.
 - If S_i is ‘1’, the machine sets b_x in the memory area of the machine. Here, b_x is the x-th element of the sequence b, if we count the elements of the sequence b so that the first element is the 0-th element.
 3. The machine shows the integer which is finally set in the memory area.
Bitaro wants to specify the string written on the stone slate. However, the number of queries cannot exceed 1000. Moreover, the maximum of the integer m input to the machine for a query should be as small as possible.
Write a program which, using the machine, specifies the string written on the stone slate.

Implementation Details
First output one line. '1' indecate a query, '0' indecate a guess.
To ask a query, first an integer m, then print two sequences separated by a space, represent sequence a and sequence b.
After flushing your output, your program should read an integer x, indicating the finally set in the memory area. You can use fflush(stdout) (if you use printf) or cout.flush() (if you use cout).
If you want to guess, output the string s. Your code should exit immediately after guessing.
Note that the answer for each test case is pre-determined. That is, the interactor is not adaptive. Also note that your guess does not count as a query.

Input
An integer N (N = 1000), represent the length of string S.

Grading
If your program is judged as correct for all the test cases, set M as the maximum of the parameter m of the function Query for all the test cases. However, if your program is judged as correct without calling the function Query, your score is calculated by M = 0.
 - If 103 <= M <= 1002,your score is 10 + ⌊(1002 - M)^2 / 9000⌋ points.
 - If 0 <= M <= 102, your score is 100 points.

Sample Communication
Assume s = "110". If we put the stone slate on the machine, input (m, a, b) = (4, [3, 3, 2, 2], [2, 2, 1, 0]),and make a query, the machine will operate as follows.
 1. The machine sets 0 in the memory area of the machine.
 2. For the first operation, since S_0 is ‘1’, the machine sets b_0, i.e. 2, in the memory area of the machine.
 3. For the second operation, since S_1 is ‘1’, the machine sets b_2, i.e. 1, in the memory area of the machine.
 4. For the third operation, since S_2 is ‘0’, the machine sets a_1, i.e. 3, in the memory area of the machine.
 5. Since the integer which is finally set in the memory area is 3, the machine shows 3.