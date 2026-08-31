Problem
Joe chose a hidden permutation p of length n consisting of all integers from 0 to n−1 (each used exactly once). 
Your goal is to recover the entire permutation.

Interactive Protocol
• You may query two distinct indices i and j (1 ≤ i, j ≤ n, i ≠ j).  
• To ask a question, print:    ? i j  
• The interactor replies with a single integer: (p_i | p_j), where | is the bitwise OR.  
• You can make at most 4269 queries(however the less queries you use, the better, as explained in the scoring section). The permutation is fixed in advance (it does not change based on your queries).

Input (Interactive Version)
• The judge first provides one line containing the integer n (3 ≤ n ≤ 2048).  
• Then you interact by printing queries as described and reading the corresponding replies.  
• If the interactor responds with −1 at any time, you either exceeded the query limit or issued an invalid query.
  You must terminate immediately after reading −1.

Output (Interactive Version)
• Once you have determined the permutation, print exactly one line:  
  ! p1 p2 … pn
• Printing the final answer does NOT count toward the 4269 query limit.

I/O & Flushing Requirements
• After every query and after printing the final answer, print a newline and flush stdout.  
  (e.g., fflush(stdout) / cout.flush() in C++; System.out.flush() in Java; stdout.flush() in Python.)

Example (Interactive Transcript)
Input
3
1
3
2

Output
? 1 2
? 1 3
? 2 3
! 1 0 2

Explanation of the Example
• The hidden permutation is [1, 0, 2].  
• Queries and replies:  
  – ? 1 2  → reply 1  (p1 | p2 = 1)  
  – ? 1 3  → reply 3  (p1 | p3 = 3)  
  – ? 2 3  → reply 2  (p2 | p3 = 2)  
• From these, you can deduce p = [1, 0, 2], then print the final answer.

Notes & Constraints
• n ranges from 3 to 2048.  
• The permutation p is over {0, 1, …, n−1}.  
• Query limit: 4269.  
• If you receive −1 at any time, terminate to avoid a wrong-answer due to reading from a closed stream.

Scoring:
• Your score for this problem is (4269-queries)/10
• Note that printing out the final answer is not counted as a query.
