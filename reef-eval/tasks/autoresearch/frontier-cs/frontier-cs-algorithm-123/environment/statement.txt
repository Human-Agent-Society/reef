Time limit per test: 1 second
Memory limit per test: 256 megabytes

Goal
-----
There is a hidden integer x with 1 ≤ x ≤ n that you must determine.

What you can do
----------------
1) Ask membership questions (up to 53 total):
   - Choose any non-empty set S ⊆ {1, 2, …, n}.
   - Ask whether x ∈ S.
   - The judge replies “YES” if x ∈ S, otherwise “NO”.

2) Make guesses (up to 2 total):
   - Output a single number as your guess for x.
   - The reply is always truthful:
     • “:)” if your guess equals x (you must terminate immediately).
     • “:(” if your guess is wrong.

Noisy answers & guarantee
--------------------------
- Not all “YES”/“NO” answers are guaranteed to be truthful.
- However, for every pair of consecutive questions, at least one answer is correct.
- This remains true across guesses: if you ask a question, then make a guess, then ask another question, the “consecutive questions” rule applies to those two questions surrounding the guess.
- Guesses themselves are always judged correctly.

Adaptive x
----------
- The judge does not fix x in advance; it may change over time.
- Changes are constrained so that all previous responses remain valid and consistent with:
  • the rule “for each two consecutive questions, at least one answer is correct,” and
  • the correctness of guess judgments.

Input
-----
- A single integer n (1 ≤ n ≤ 100000), the maximum possible value of x.

Interactive protocol (I/O format)
----------------------------------
To ask a question about a set S:
- Print a line: “? k s1 s2 … sk”
  • k = |S| (k ≥ 1)
  • s1, s2, …, sk are distinct integers in [1, n]
- Flush output immediately.
- Read a single word reply: “YES” or “NO”.

To make a guess for x:
- Print a line: “! g” where g is your guess (1 ≤ g ≤ n).
- Flush output immediately.
- Read the judge’s reply:
  • “:)” if correct — your program must terminate immediately.
  • “:(” if incorrect — you may continue if you still have remaining queries/guesses.

Flushing
--------
After every printed line, flush the output to avoid idleness/timeout:
- C++: fflush(stdout) or cout.flush()
- Java: System.out.flush()
- Pascal: flush(output)
- Python: sys.stdout.flush()
- See language docs for others.

Limits
------
- Maximum questions: 53
- Maximum guesses: 2
- n up to 100000

Important notes
----------------
- Because at least one of every two consecutive question answers is correct, you can design strategies that compare adjacent answers to filter lies.
- Guesses are always reliable; use them sparingly (you only have 2).
- Note that this problem has a scoring system. You are graded based on the # of queries you use. The lower the # of queries you use, the higher the score you get.

Example
--------
Input
  6

(Sequence of interactions as seen by the contestant)
  ? 5 1 2 5 4 3
  NO
  ! 6
  :(
  ? 4 1 2 3 4
  NO
  ! 5
  :)

Explanation
- If the first question’s “NO” had been truthful, x would have to be 6.
- The guess “! 6” receives “:(“, so 6 is not the answer. Therefore, the first answer must have been a lie.
- By the guarantee, the next question’s answer must then be truthful.
- From “? 4 1 2 3 4” with reply “NO”, we conclude x ∉ {1,2,3,4}; combined with “6 is wrong”, x must be 5.
- The final guess “! 5” is confirmed with “:)”.
