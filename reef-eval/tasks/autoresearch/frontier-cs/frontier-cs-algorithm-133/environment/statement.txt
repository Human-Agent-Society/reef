Time limit:20 second
Memory limit:1024 megabytes
One day, Little Z selected n points on a piece of paper and connected them pairwise with a pencil to form n*(n-1)/2 line segments. Due to the thinness of the pencil, it can be assumed that the width of these line segments is 0.
Looking at these line segments, Little Z fell into meditation. He believes that some of these line segments are important and need to be emphasized. Therefore, Little Z took out a brush and stroked them. A brush drawn on paper will form a circle with a circle with radius r. When stroking a line segment, the center of the brush (i.e. the center of the circle) will start from one endpoint of the line segment and trace along the line segment towards the other endpoint.
Now, Little Z really wants to know how much area on the paper is emphasized after the stroke. Can you help him answer this question?

Input
The first line of the input file is a positive integer n (n <= 2410), representing the number of selected points.
From the second line to the n+1 line, where the i+1 line contains two real numbers x_i and y_i (|x_i|,|y_i| <= 100.0), representing the coordinates of point i as (x_i, y_i).
The n+2 line is a positive integer m (m <= n*(n-1)/2) , representing the number of line segments that Little Z considers important.
From line n+3 to line n+m+2, each line has two positive integers a, b (1 <= a, b <= n), representing a line segment. Among them, a and b represent the numbering of the two endpoints of the line segment, respectively.
In the n+m+3 line, there is a real number r (0 < r <= 5.0) representing the radius of the circle formed by the brush on the paper.
In line n+m+4 , there are four real numbers p_1, p_2, p_3, p_4, which are the parameters used for scoring.

Output
Only one number per line represents the total area of the highlighted area after the stroke.

Example
standard input
2
1 1
1 2
1
1 2
1
0.00001 0.001 0.1 1

standard output
5.1415927

Scoring

Note:
This problem provides four parameters p1, p2, p3, p4 in the input file.
All four values must be read from the input, but only p1 and p3 are used for scoring;
p2 and p4 are ignored.

Let difference denote the absolute difference between your answer and the standard answer.
Your score for each test point is computed as follows:

score = 100 / (1 + p1 * p3)


The total score is the sum of the scores over all test points.
