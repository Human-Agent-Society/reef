Problem: Matrix k-th Smallest (Interactive)

You are given an unknown n×n matrix a. You must find the k-th smallest value among all a[i][j].
You cannot read the matrix directly. Instead, you must interact with the judge using the protocol below.
The matrix satisfies:
for all i<=n and j<n: a{i,j}<=a{i,j+1}
for all i<n and j<=n: a{i,j}<=a{i+1,j}

- First you need to read the input n and k.
- Your program must write commands to stdout and read responses from stdin. Flush after every command.

Supported commands you may send:
1) QUERY x y
   - Asks for the value a[x][y].
   - Constraints: 1 ≤ x, y ≤ n.
   - The interactor replies with an integer v:
     v
     where v = a[x][y].

2) DONE ans
   - You announce your final answer.
   - The interactor will terminate after printing a single floating-point score to stdout.

Limits
- You may call QUERY at most 50000	times per test file.
- Any out-of-bounds query or exceeding the query limit results in score 0.0.
- n<=2000, a{i,j}<=10^18
- You need to use c++17
- Time limit: 5 seconds per test.
- Memory limit: 1024 MB.

Scoring
- Let used be the number of QUERY calls you made for the current test.
- Let correct be the true k-th smallest value among all 50000 entries.
- If your final ans equals correct, your score is:
    if used ≤ n: 1.0
    else if used ≥ n*n: 0.0
    else: (50000 - used) / (50000 - n)
  Otherwise, your score is 0.0.
