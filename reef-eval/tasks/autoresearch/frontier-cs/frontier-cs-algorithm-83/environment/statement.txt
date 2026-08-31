# Function

## Problem Description

Construct a **completely multiplicative function** \( f \) (that is, \( f(xy) = f(x)f(y) \)) such that \(|f(i)| = 1\) for all \(i\), and **minimize**

\[
\max_{1 \le k \le n} \left| \sum_{i=1}^k f(i) \right|.
\]

## Input

A single integer \( n \).

## Output

Output \( n \) integers on one line, representing \( f(1), f(2), \ldots, f(n) \).

## Constraints

There is only **one test case**, with \( n = 10^6 \).

## Scoring

Let your output be `out`, and the standard answer be `ans`.  
Your score is computed based on the difference between `out` and `ans` (exact formula not specified in the statement).
