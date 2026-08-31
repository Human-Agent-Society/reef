SphereSpread

You are given an integer n. You need to place n points in 3D space such that all points lie within or on 
a unit sphere centered at the origin (i.e., the distance from each point to the origin is at most 1).

Your goal is to maximize the minimum pairwise distance between any two points. In other words, you want 
to spread the points out as much as possible, maximizing the distance between the closest pair.

Input
The first line contains a single integer n — the number of points to place.

Output
On the first line, print a single real number min_dist — the minimum pairwise distance achieved by your 
point placement.
The next n lines should each contain three real numbers xi, yi, zi — the coordinates of the i-th point.

All coordinates must satisfy xi² + yi² + zi² ≤ 1 (the point lies within or on the unit sphere).

Constraints
2 <= n <= 1000

Your answer will be accepted if:
- All points are within or on the unit sphere (with absolute or relative error at most 10^-9)
- The actual minimum pairwise distance matches your claimed min_dist (with absolute or relative error at most 10^-6)

Scoring
You will be graded based on the minimum pairwise distances you achieve.
To be more specific, your answer will be compared to a reference solution ref_answer.
Your final score will be calculated as the average of 100 * min(your_answer / ref_answer, 1) across all test cases.

Time limit: 2 seconds

Memory limit: 512 MB

Sample Input:
2

Sample Output:
2
0 0 1
0 0 -1