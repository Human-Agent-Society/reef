# AHC058: Apple Production Planning

Write a C++20 program that selects investments over `T` turns to maximize
apple production. There are `L` machine levels and `N` machine IDs.
Machine `(i, j)` has count `B[i,j]`, power `P[i,j]`, and base upgrade cost `C[i,j]`.
Initially every count is 1, every power is 0, and the apple balance is `K`.

At each turn, select one action:

- Upgrade `(i, j)`: pay `C[i,j] * (P[i,j] + 1)` apples, then increase its power by 1.
- Do nothing.

An upgrade is illegal if its cost exceeds the current apple balance.
After the action, production occurs in increasing level order:

1. Level 0 adds `A[j] * B[0,j] * P[0,j]` apples for every ID `j`.
2. Level 1 adds `B[1,j] * P[1,j]` to `B[0,j]`.
3. Level 2 adds `B[2,j] * P[2,j]` to `B[1,j]`.
4. Level 3 adds `B[3,j] * P[3,j]` to `B[2,j]`.

The order matters: new lower-level machines contribute from the next turn.
For final apple balance `S`, the case score is `round(100000 * log2(S))`.
The trusted public evaluator runs 150 cases. The search objective maximizes
their mean raw score, and every case must pass before a program enters the archive.

## Input

Read the following text from standard input:

```text
N L T K
A[0] A[1] ... A[N-1]
C[0,0] C[0,1] ... C[0,N-1]
...
C[L-1,0] C[L-1,1] ... C[L-1,N-1]
```

The benchmark uses `N=10`, `L=4`, `T=500`, and `K=1`.
Productivities and costs are positive integers. `A[0]=1`, and productivities
are in ascending order. `C[0,0]=1`.

## Output

Write exactly `T` actions to standard output. An upgrade is one line `i j`.
A no-op is one line `-1`. Lines beginning with `#` are comments.
Use arithmetic that handles the growing apple and machine counts without overflow.
The evaluator enforces a two-second limit and 1 GiB memory per case.
Do not access the evaluator, external services, or saved answers.
Return the complete program in a fenced `cpp` block inside `<solution>`.
