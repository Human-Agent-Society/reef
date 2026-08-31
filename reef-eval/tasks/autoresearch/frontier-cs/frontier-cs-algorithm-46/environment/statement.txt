Title: Job Shop Scheduling (JSPLIB-style) — Open Optimization Track

Overview
--------
You are given a classic Job Shop Scheduling Problem (JSSP). There are J jobs and M machines.
Each job must be processed exactly once on each machine, in a job-specific order (its *route*).
Processing is non-preemptive. A machine can process at most one operation at a time. The goal is to minimize the *makespan*:
the completion time of the last operation among all jobs.

This problem is NP-hard. We therefore use an *open scoring* scheme that rewards better (lower) makespans. See **Scoring** below.

Terminology
-----------
• Operation: A single (job, machine) processing step with a fixed processing time.  
• Route of a job j: A sequence of M distinct machines (0..M-1) listing the order in which job j must visit them.  
• Precedence (job chain): If the k-th operation of job j precedes its (k+1)-th, the latter cannot start before the former finishes.  
• Resource constraint (machine): Operations assigned to the same machine cannot overlap in time.  
• Makespan (C_max): The maximum completion time over all operations in the schedule.

Input Format
------------
The input is plain text with 0-based indices.

Line 1:
  J M
  • J (integer): number of jobs (J ≥ 1)
  • M (integer): number of machines (M ≥ 1)

Lines 2..(J+1): one line per job j in order j = 0..J-1. Each line contains 2*M integers
representing the route and processing times for job j:

  m_0 p_0  m_1 p_1  ...  m_{M-1} p_{M-1}

where:
  • m_k ∈ {0,1,...,M-1} is the machine index of the k-th operation of job j.  
  • p_k is a positive integer (processing time of that operation).  
  • Each machine index must appear **exactly once** in a job’s line (every job uses every machine exactly once).  
  • The order of the pairs on the line determines the job’s precedence constraints.

Output Format
-------------
You must output **exactly M lines**.
Line m (for m = 0..M-1) must contain **J distinct integers**: a permutation of {0,1,...,J-1}.  
This permutation specifies the order in which machine m processes the J jobs (from first to last).

Important:
• You **do not** print start or finish times.  
• Your permutations must mention each job exactly once on every machine. Otherwise, the checker will reject the output.  
• The judge constructs the earliest-feasible schedule implied by your machine orders and the job precedence constraints
  (equivalently: the longest-path length in the disjunctive graph with your chosen orientations on machine arcs).

Validity Rules
--------------
Your output is rejected if any of the following occurs:
• A machine line does not contain a permutation of {0..J-1} (duplicate/missing job index; out-of-range index).  
• The machine orders together with the job precedence constraints induce a cycle in the disjunctive graph
  (i.e., there exists no feasible schedule consistent with your machine orders).

Scoring (Lower is Better)
-------------------------
Let P be the makespan computed from your output for a test case. The answer file for each test contains two integers:
(B, T). For this problem, **T is fixed to 0** (a trivial lower bound), and **B > 0** is the makespan of a simple feasible
baseline schedule (a naïve dispatch heuristic). The checker applies the general formula:

  If B ≤ T:   score = 1.0 if P ≤ T else 0.0
  Else:       score = clamp( (B - P) / (B - T), 0, 1 )

With T = 0 and B > 0, this simplifies to:

  score = clamp( 1 - P / B, 0, 1 )

Your final problem score is the average of your per-test scores. The checker prints partial credit messages containing
the substring “Ratio: <value>” as required by the judge.

Constraints
-----------
The official test set is deliberately challenging:
• Sizes range up to ~ (J, M) ≈ (50, 25) (total operations up to ~1,250).  
• Processing times are positive integers and may range broadly (including very large values).  
• Route structures include random, nearly-flow, block-flow, and strong bottlenecks (one or two machines dominating).

Tips
----
Feasible and competitive schedules often come from combinations of:
• Priority rules (SPT/LPT/weighted), bottleneck-aware dispatching.  
• Local improvement via adjacent swaps per machine.  
• Metaheuristics (tabu search, simulated annealing, iterated local search).  
• Shifting bottleneck heuristics or relax-and-fix styles.

Example (Illustrative Only; Not in the Tests)
---------------------------------------------
Input:
  3 2
  0 3  1 4
  1 2  0 5
  0 4  1 1

Valid Output (two lines; each line is a permutation of {0,1,2}):
  2 0 1
  1 2 0

This tells the judge to process jobs on machine 0 in order [2,0,1] and on machine 1 in order [1,2,0].
The judge then computes the earliest-feasible schedule and its makespan.
