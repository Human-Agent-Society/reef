Operators

-   You are given n operators op₁, op₂, …, opₙ. For an input sequence of
    n+1 integers a₀, a₁, …, aₙ, you need to compute the following value:
    (((…(a₀ op₁ a₁) op₂ a₂) op₃ a₃)… opₙ aₙ) mod (10⁹ + 7).
-   For any 0 ≤ i ≤ n, we have 0 ≤ aᵢ < 10⁹ + 7.
-   For any 1 ≤ i ≤ n, opᵢ ∈ {+, ×}; that is, each operator is either
    addition “+” or multiplication “×”.

You quickly wrote a program to solve this problem. Unfortunately,
because you were over‑excited after solving it, you accidentally deleted
the original problem statement and your code. As a result, you lost the
information about the n operators op₁, …, opₙ. Fortunately, you still
have the program you just compiled, so you can recover these operators
by querying your program.

Since the contest is about to end, and the program you wrote is not
efficient enough, you cannot ask more than Qₘₐₓ = 600 queries.

Implementation details

This is an interactive problem. As usual, you should submit a source code file that can compile.  

Initially, you should read an integer n, the number of operators.

You may issue queries by writing to standard output lines of the form  

? a₀ a₁ … aₙ

1 <= a_i < 10^9+7

Then you must flush output, and read from standard input one integer in [0, 10⁹ + 6], representing the interactor’s response.

When you have determined the operators, output a line of the form

! o₁ o₂ … oₙ

o_i is 0 if op_i is "+", is 1 if op_i is "×".

Remember to flush after your output.

Subtasks

-   For all data, 1 ≤ n ≤ 600.
-   Let your program make Q queries:
    -   If your program exceeds time limit, memory limit, or returns
        incorrect answer → score=0.
    -   Otherwise, your score depends on Q:
        -   score(Q) = 41 / (Q + 1)
        -   In other words, a solution with Q ≤ 40 is awarded the full score.