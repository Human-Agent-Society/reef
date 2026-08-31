Problem Statement

There are n vertices forming a simple cycle. It is guaranteed that the graph satisfies the following property:

- There exists a permutation p of length n such that for every i (1 ≤ i < n), the vertices p_i and p_{i+1} are adjacent in the graph, and also p_n and p_1 are adjacent.

At the beginning of the interaction, a token is placed at a predetermined starting vertex s (1 ≤ s ≤ n).  
The value of n is hidden from the contestant. The contestant may interact with the judge in order to determine n.

---
Interaction Protocol

The contestant may issue the following commands:

1. walk x
   - Input: a nonnegative integer x (0 ≤ x ≤ 10^9).
   - Effect: the token moves x steps forward along the cycle from its current position.
   - Output: the label of the vertex reached after the move.

2. guess g
   - Input: an integer g (1 ≤ g ≤ 10^9).
   - Effect: ends the interaction.
   - If g = n, the answer is considered correct. Otherwise, it is considered wrong.

Constraints:
- 1 ≤ n ≤ 10^9
- 1 ≤ s ≤ n
- The number of walk operations must not exceed 200000.

---
Scoring

Let q be the number of walk operations made before the guess.

- If the guess is wrong or q > 200000, the score is 0.
- Otherwise, the score is f(q), where f is a continuous, monotonically decreasing function defined in log10-space by linear interpolation through the following anchor points:

  f(1)     = 100  
  f(10000) = 95  
  f(20000) = 60  
  f(50000) = 30  

and further decreasing linearly to

  f(200000) = 0.

Thus, fewer walk queries yield a higher score, with a perfect solution scoring close to 100.
