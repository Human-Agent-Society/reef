Interactive RBS

This is an I/O interactive problem. I/O interaction refers to interactive problems, where the program communicates with a special judge during execution instead of producing all output at once. In these problems, the program sends queries (output) to the judge and must immediately read responses (input) before continuing. The solution must strictly follow the input-output protocol defined in the problem statement, because any extra output, missing flush, or incorrect format can cause a wrong answer. Unlike standard problems, interactive problems require careful handling of I/O, synchronization, and flushing to ensure smooth communication between the contestant’s code and the judge.

There is a hidden bracket sequence s of length n, where s only contains '(' and ')'. It is guaranteed that s contains at least one '(' and one ')'.

To find this bracket sequence, you can ask queries. Each query has the following form: you pick an integer k and arbitrary indices i_1,i_2,...,i_k (1<=k<=1000, 1<=i_1,i_2,...,i_k<=n). Note that the indices can be equal. Next, you receive an integer f(s_{i_1}s_{i_2}...s_{i_k}) calculated by the jury.

For a bracket sequence t, f(t) is the number of non-empty regular bracket substrings in t (the substrings must be contiguous). For example, f("()())")=3.

A bracket sequence is called regular if it can be constructed in the following way.

1. The empty sequence ∅ is regular.
2. If the bracket sequence A is regular, then (A) is also regular.
3. If the bracket sequences A and B are regular, then the concatenated sequence AB is also regular.

For example, the sequences "(())()", "()" are regular, while "(()" and "())(" are not.

Find the sequence s using no more than 200 queries. Specifically, your score will be (200 - q) / 200, where q is the number of queries.

The first line of each test case contains one integer n (2<=n<=1000). At this moment, the bracket sequence s is chosen. The interactor in this task is not adaptive. In other words, the bracket sequence s is fixed in every test case and does not change during the interaction.

To ask a query, you need to pick an integer k and arbitrary indices i_1,i_2,...,i_k (1<=k<=1000), 1<=i_1,i_2,...,i_k<=n) and print the line of the following form (without quotes):

"0 k i_1 i_2 ... i_k"

After that, you receive an integer f(s_{i_1}s_{i_2}...s_{i_k}).

You can ask at most 200 queries of this form.

Next, if your program has found the bracket sequence s, print a line with the following format (without quotes):

"1 s_1s_2...s_n"

Note that this line is not considered a query and is not taken into account when counting the number of queries asked.

If you ask more than 200 queries during an interaction, your program must terminate immediately, and you will receive the Wrong Answer verdict. Otherwise, you can get an arbitrary verdict because your solution will continue to read from a closed stream.

After printing a query or the answer for a test case, do not forget to output the end of the line and flush the output. Otherwise, you will get the verdict Idleness Limit Exceeded. To do this, use:

fflush(stdout) or cout.flush() in C++;
System.out.flush() in Java;
flush(output) in Pascal;
stdout.flush() in Python;
see the documentation for other languages.

Example Input 1 (interactor to you):

3

0

1

1

Example Output 1 (you to interactor):


0 4 1 2 3 3

0 2 2 1

0 2 3 1

1 )((

Example Input 2 (interactor to you):

2

3

Example Output 2 (you to interactor):


0 4 1 2 1 2

1 ()