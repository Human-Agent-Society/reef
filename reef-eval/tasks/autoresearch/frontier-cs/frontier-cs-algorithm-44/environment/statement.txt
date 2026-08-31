Problem: Traveling Santa with Carrot Constraint

Story
Rudolph plans to shorten Santa’s route by choosing a better order to visit cities. Every 10th step takes 10% longer unless that step starts from a prime-numbered city. Your task is to output a valid tour and compete on how much you improve over a strengthened baseline.

Formal definition
- There are N cities labeled 0, 1, 2, …, N−1. City 0 is the North Pole.
- Each city i has 2D Cartesian coordinates (xi, yi).
- A route is a sequence P of length N+1: P0, P1, …, PN with:
  - P0 = PN = 0
  - {P1, P2, …, PN−1} is a permutation of {1, 2, …, N−1}
- The Euclidean distance between cities a and b is dist(a, b) = sqrt((xa − xb)^2 + (ya − yb)^2).
- Carrot constraint (10% step penalty):
  - For each step t = 1, 2, …, N (moving from P[t−1] to P[t]), define a multiplier:
    - If t is a multiple of 10 and P[t−1] is not a prime number, multiplier m[t] = 1.1
    - Otherwise, m[t] = 1.0
  - We use the standard definition of primes over city IDs: 2, 3, 5, 7, 11, … are prime; 0 and 1 are not prime.
- The penalized length of the route is:
  L(P) = sum over t = 1..N of m[t] × dist(P[t−1], P[t])

Goal
Minimize L(P) subject to the route validity constraints.

Input
- First line: integer N (2 ≤ N ≤ 200000).
- Next N lines: two integers xi yi for i = 0..N−1.
  - Coordinates satisfy |xi|, |yi| ≤ 10^9.
  - IMPORTANT: The N lines are given in strictly increasing order by x:
    x0 < x1 < x2 < … < xN−1.
    City IDs equal their input order. This strengthens the baseline path that follows the input order.

Output
- First line: integer K, which must be exactly N+1.
- Next K lines: one integer per line, the city ID sequence P0, P1, …, PN.
  - Must satisfy P0 = PN = 0 and visit every city 1..N−1 exactly once.

Validity checks
- City IDs must be in [0, N−1].
- The sequence must start and end at 0.
- Each city 1..N−1 must appear exactly once between P1 and PN−1.
- If any of these fail, the output is invalid and receives 0 for that test.

Notes and clarifications
- Step indexing for penalties is global over the entire tour: steps t = 10, 20, 30, … may be penalized depending on the source city ID at that step.
- If N < 10, no step index is a multiple of 10, so no 10% penalties occur.
- 0 and 1 are not prime. The first prime city ID is 2.
- Distances and sums are computed in double precision; no rounding beyond floating-point arithmetic.
- The platform may display normalized scores in [0, 100]; the checker outputs [0, 1] (after remap).
- Multiple tests of varying sizes are used; your final score is the average of per-test scores.

Constraints and limits
- Time limit: 2 seconds
- Memory limit: 512 MB

Sample
Input
5
0 0
1 0
2 0
3 1
4 1

Output
6
0
1
2
3
4
0

(The sample illustrates format only.)
