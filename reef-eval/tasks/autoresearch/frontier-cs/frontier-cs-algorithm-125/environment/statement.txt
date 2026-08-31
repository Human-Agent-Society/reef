Time Limit: 1 second
Memory Limit: 256 MB

Overview
This is an INTERACTIVE PROBLEM. There are N kinds of minerals. For each kind there are exactly 2 slices, for a total of 2N slices numbered 1..2N. The judge fixes a hidden pairing: for each kind, the two slices of that kind form one pair. Your task is to determine all N pairs.

You have access to a device. You may insert or extract slices from the device one at a time. After each operation, you learn how many DISTINCT kinds of minerals are currently present among the slices inside the device.

Goal
Determine all N pairs while MINIMIZING the number of queries you use. You may use at most 1,000,000 queries. Any correct solution using ≤ 1,000,000 queries is accepted; fewer queries are considered better.

Interaction Protocol (standard input/output)
1) At the start, the judge outputs a single integer:
   N
   • 1 ≤ N ≤ 43,000.

2) You may then repeatedly perform queries. To toggle the presence of slice x in the device:
   • Output a line:  ? x
     where 1 ≤ x ≤ 2N
   • Flush stdout.
   • Read a single integer r from stdin.
     r is the number of DISTINCT kinds currently present among all slices inside the device after this toggle:
       – If x was not in the device, it is now inserted.
       – If x was already in the device, it is extracted.
       – r counts how many mineral kinds appear at least once among the slices currently in the device.

3) When you have determined a pair (a, b), output exactly one line:
   • Output:  ! a b
     where 1 ≤ a ≤ 2N and 1 ≤ b ≤ 2N.
   Over the entire run you must output exactly N such lines, and together they must use each index 1..2N exactly once.

4) Order is flexible:
   • You may interleave “? x” queries and “! a b” answers in any order.
   • The judge terminates the interaction immediately after reading the N-th valid “! a b” line. Do not send any further output after that point.

Important Rules and Constraints
• Only print lines of the form “? x” and “! a b”.
• Indices in queries and answers must satisfy their ranges.
• Exactly N answer lines must be printed and together cover each index 1..2N exactly once.
• A “query” is defined as one printed line “? x”. You may perform at most 1,000,000 queries.
• Flush stdout after every line you print (interactive).
• If you violate the protocol (bad format, invalid index, wrong pairings, too many queries, wrong number of answers), the judge will return a Wrong Answer verdict with a message.

Device Behavior (for clarity)
• The device maintains a set S of slices currently inside.
• Query “? x” toggles membership of x in S:
  – If x ∉ S, insert x.
  – Else (x ∈ S), remove x.
• The judge replies with r = number of DISTINCT mineral kinds represented by S. If S is empty, r = 0.

Scoring / Ratio (informative)
• Let Q be your total number of “? x” queries.
• The judge also knows an optimal_queries value for the instance.
• Your ratio is (1,000,000 − Q) / (1,000,000 − optimal_queries).
• The judge reports this ratio and scores only on Accepted submissions.

Sample Communication
Judge → program:
4

Program → judge:
? 1
Judge → program:
1

Program → judge:
? 2
Judge → program:
2

Program → judge:
? 5
Judge → program:
2

Program → judge:
? 2
Judge → program:
1

Program → judge:
! 3 4
Program → judge:
! 5 1
Program → judge:
! 8 7
Program → judge:
! 2 6

(Here, the program used 4 queries total.)
