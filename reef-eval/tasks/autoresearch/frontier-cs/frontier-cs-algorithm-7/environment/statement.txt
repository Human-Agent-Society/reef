Build a Computer
Input ﬁle: standard input
Output ﬁle: standard output
Time limit: 1 second
Memory limit: 1024 megabytes
You want to build a computer to achieve a speciﬁc functionality: Given an integerx, determine whether
x lies within the interval[L, R]. To accomplish this, you designed a directed acyclic graph (DAG) with
edge weights of0 and 1, which contains a starting node with an indegree of0 and an ending node with an
outdegree of0. By starting from the starting node and following a path to the ending node, the sequence
of the traversed edge weights forms a binary representation of an integer within the range[L, R] without
leading zeros. Every integer within the range[L, R] must correspond to exactly one unique path in this
graph. In this way, you can determine whether an integer lies within the range[L, R] by checking if its
binary representation can be constructed by traversing this DAG.
Clearly, you could separate the corresponding path for each integer into individual chains. However, you
realized that for a large range, such a DAG would require too many nodes, and the computer you built with
only 256 MiB of memory cannot store it. Therefore, you need to compress this DAG, allowing diﬀerent
paths to share nodes, in order to reduce the number of nodes and edges. Formally, you need to construct
a DAG with no more than100 nodes, where each node has an outdegree of at most200. The DAG must
have edge weights of0 and 1, with exactly one starting node with an in-degree of0 and one ending node
with an out-degree of0. Every integer in the range[L, R] must correspond toexactly one unique path
from the start to the end in this DAG, and no path should represent any integer outside the range[L, R].
Note that none of the binary sequences formed by any path in the graph should have leading zeros. There
may be two edges with diﬀerent weights between two nodes.
Input
A single line containing two positive integersL, R(1 ≤L ≤R ≤106).
Output
The ﬁrst line should output the number of nodesn (1 ≤n ≤100). You need to make n as small as possible and your score will be determined by the value of n.
For the nextn lines, thei-th line should start with an integerk (0 ≤k ≤200), representing the number
of outgoing edges from nodei. Then output2 ·k integers ai,k, vi,k (1 ≤ai,k ≤n, ai,k ̸= i, vi,k ∈{0, 1}),
which means that nodei has a directed edge with weightvi,k to node ai,k. You must ensure that the
output represents a directed acyclic graph that satisﬁes the requirements.
Example
standard input standard output
5 7 8
3 2 1 3 1 4 1
1 5 0
1 6 1
1 7 1
1 8 1
1 8 0
1 8 1
0
Page 1 of 1