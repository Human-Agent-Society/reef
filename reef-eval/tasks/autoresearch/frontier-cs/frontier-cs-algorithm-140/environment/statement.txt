Mineral Deposits (interactive)

You handle signal processing for an extra-terrestrial mining company. Your vessel is approaching an asteroid.
Preliminary scans show the presence of k mineral deposits on the asteroid, but their precise locations are unknown.

The surface of the asteroid is modeled as a grid of integer coordinates. Each mineral deposit i is at unknown integer
coordinates (x_i, y_i) with −b ≤ x_i ≤ b and −b ≤ y_i ≤ b, for some integer b corresponding to the size of your initial scan.

You may send probes to the surface in waves.

If you send a wave of d probes at coordinates (s_j, t_j) for j = 1..d, then when a probe arrives, it measures the Manhattan
distance to each of the k deposits. All data packets arrive together and are indistinguishable across probes.
Thus, one wave returns the k·d integer distances:
  |x_i − s_j| + |y_i − t_j|  for all i in {1..k} and j in {1..d}.
The list of returned distances is in non-decreasing order.

Goal
— Minimize the number of probe waves needed to determine all deposit locations.

Interaction
At the start, read a single line containing three integers: b, k, w — the scan boundary, the number of deposits,
and the maximum number of waves you may send.

You may then make at most w queries, each representing one wave. A query is printed as:
  ? d s1 t1 s2 t2 ... sd td
with 1 ≤ d ≤ 2000. Each probe coordinate must satisfy −10^8 ≤ s_j, t_j ≤ 10^8.
The judge replies with one line containing k·d integers in non-decreasing order: the multiset of all Manhattan
distances between the deposits and the d probe coordinates.

The total number of probes across all waves must not exceed 2·10^4.

To finish, print one line:
  ! x1 y1 x2 y2 ... xk yk
containing the coordinates of all k deposits in any order. This must be your last line of output.

Base Constraints
1 ≤ b ≤ 10^8, 1 ≤ k ≤ 20, and 2 ≤ w ≤ 10^4.

Scoring
For each test case, your score is:
  (# of mineral deposits found) / k
Your overall score is the average over all test cases. There are no point-based subtasks in this version.

Example
If k = 2 deposits are at (1, 2) and (−3, −2), and you send d = 3 probes to (−4, −3), (−1, 0), and (2, −1),
you must print:
  ? 3 -4 -3 -1 0 2 -1
and the response would be the six distances:
  2 4 4 4 6 10
If the next wave has d = 2 probes at (1, 2) and (0, −2), you must print:
  ? 2 1 2 0 -2
and the response would be:
  0 3 5 8
Finally you might answer:
  ! 1 2 −3 −2

Implementation notes:
You may not ask more than w queries. Once you ask w queries and do the respective calculations, you should just print out the locations of the mineral deposits.