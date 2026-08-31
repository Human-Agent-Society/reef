Time limit per test: 10 seconds
Memory limit per test: 256 megabytes
Note: The only difference between versions of this problem is the maximum value of n.

Overview
Professor Vector is preparing to teach her Arithmancy class. She needs to prepare n distinct magic words for the class. Each magic word is a string over the alphabet {X, O}. A spell is created by concatenating two magic words. The power of a spell is defined as the number of distinct non-empty substrings of the resulting string.

Example: The power of the spell "XOXO" is 7, because it has 7 distinct non-empty substrings:
X, O, XO, OX, XOX, OXO, XOXO.

Task Summary
Your program must:
1) Read n and output n distinct magic words (w1, w2, ..., wn).
2) Then read q (the number of students/queries).
3) For each student j = 1..q:
   - Read pj, the power of the student’s spell.
   - Output the exact pair of indices (uj, vj) such that concatenating w_uj followed by w_vj produces a string whose power is pj.
   - The order matters: you must output the indices in the correct order (first word, then second word).

Interaction Protocol (Interactive Problem)
1) Input: A single integer n (1 ≤ n ≤ 1000), the number of magic words to prepare.
2) Output: Print n distinct magic words, one per line.
   - Each magic word must:
     • Consist only of characters 'X' and 'O'.
     • Have length between 1 and 30·n (inclusive).
   - Denote the i-th printed word as w_i (1 ≤ i ≤ n).
   - After printing all n words, flush the output.

3) Input: A single integer q (1 ≤ q ≤ 1000), the number of students.
4) For each of the q students:
   - Input: A single integer pj, the power of their spell.
     • It is guaranteed that pj was produced as follows:
       ◦ Two indices uj and vj were chosen independently and uniformly at random from {1, 2, ..., n}.
       ◦ The spell S was formed by concatenating w_uj and w_vj (in this order).
       ◦ pj equals the number of distinct non-empty substrings of S.
   - Output: Print the two integers uj and vj (1 ≤ uj, vj ≤ n) in this order.
   - Flush after printing each pair (uj, vj).

Important Requirements and Notes
• Distinctness: All n magic words you output must be distinct.
• Alphabet: Only 'X' and 'O' are allowed in each magic word.
• Length bounds: For each word, 1 ≤ |w_i| ≤ 30·n.
• Exact identification: For each query, it is not enough to find any pair that yields pj. You must identify the exact two words used by the student and the correct order (w_uj then w_vj).
• Flushing: Remember to flush after:
  – Printing all n words.
  – Printing each answer pair (uj, vj).

Definitions
• Magic word: A string over {X, O}.
• Spell: Concatenation of two magic words (w_a followed by w_b).
• Power of a string: The number of different non-empty substrings of that string.

Constraints
• 1 ≤ n ≤ 1000
• 1 ≤ q ≤ 1000
• For each i: 1 ≤ |w_i| ≤ 30·n
• Alphabet: {X, O} only

Example
Input (conceptual, since the problem is interactive):
2
2
15
11

Output (one possible valid interaction transcript):
XOXO
X
1 1
2 1

Explanation of the example:
• After reading n = 2, the program outputs two distinct magic words, e.g., w1 = "XOXO" and w2 = "X" (flush).
• Then q = 2 is read.
• For the first student, the power p1 = 15 is read; the program answers with the indices "1 1".
• For the second student, the power p2 = 11 is read; the program answers with the indices "2 1".
(Exact details of powers and indices depend on the judge’s random choices and the specific words you output.)

Scoring:
Your score for this problem is not just an AC or WA verdict. The scoring is based on the following formula:
(30n^2 - Total length of your magic words)/(30n^2 - Optimal total length of magic words)
Your goal is to maximize your score.
