time limit per test: 1 second
memory limit per test: 256 megabytes
This is an interactive problem.
Zookeeper has set up a special lock for the rabbit enclosure.
The lock consists of n concentric rings numbered from 0 to n−1. The innermost ring is ring 0 and the outermost ring is ring n−1. All rings are split equally into n*m sections each. Each of those rings contains a single metal arc that covers exactly m contiguous sections. At the center of the ring is a core and surrounding the entire lock are n*m receivers aligned to the n*m sections.
The core has n*m lasers that shine outward from the center, one for each section. The lasers can be blocked by any of the arcs. A display on the outside of the lock shows how many lasers hit the outer receivers.
For example, there are n=3 rings, each covering m=4 sections. There are n*m=12 sections. The ring 0 covers sections 0, 1, 2, 3, the ring 1 covers sections 1, 2, 3, 4, and ring 2 covers sections 7, 8, 9, 10. Three of the lasers (sections 5, 6, 11) are not blocked by any arc, thus the display will show 3 in this case.
Wabbit cannot see where any of the arcs are. Given the relative positions of the arcs, Wabbit can open the lock on his own.
To be precise, Wabbit needs n−1 integers p_1,p_2,…,p_{n−1} satisfying 0≤p_i<n*m such that for each i (1≤i<n), Wabbit can rotate ring 0 clockwise exactly p_i times such that the sections that ring 0 covers perfectly aligns with the sections that ring i covers. In the example above, the relative positions are p_1=1 and p_2=7.
To operate the lock, he can pick any of the n rings and rotate them by 1 section either clockwise or anti-clockwise. You will see the number on the display after every rotation.
Wabbit has asked you to help him to find the relative positions of the arcs after all of your rotations are completed. You may perform up to 30000 rotations before Wabbit gets impatient, and your score is inversely linear related to the max number of queries.

Input
The first line consists of 2 integers n and m (2≤n≤100,2≤m≤20), indicating the number of rings and the number of sections each ring covers.

Interaction
To perform a rotation, print on a single line "? x d" where x (0≤x<n) is the ring that you wish to rotate and d (d∈{−1,1}) is the direction that you would like to rotate in. d=1 indicates a clockwise rotation by 1 section while d=−1 indicates an anticlockwise rotation by 1 section.
For each query, you will receive a single integer a: the number of lasers that are not blocked by any of the arcs after the rotation has been performed.
Once you have figured out the relative positions of the arcs, print ! followed by n−1 integers p_1,p_2,…,p_{n−1}.
Do note that the positions of the rings are predetermined for each test case and won't change during the interaction process.
After printing a query do not forget to output the end of line and flush the output. Use fflush(stdout) or cout.flush() to flush.

Example
Input
3 4
4
4
3
Output
? 0 1
? 2 -1
? 1 1
! 1 5

Note
For the first test, the configuration is the same as the example form the statement.
After the first rotation (which is rotating ring 0 clockwise by 1 section), we have the ring 0 covers sections 1, 2, 3, 4, the ring 1 covers sections 1, 2, 3, 4, and ring 2 covers sections 7, 8, 9, 10. The sections 0, 5, 6, 11 are not blocked by any arc.
After the second rotation (which is rotating ring 2 counter-clockwise by 1 section), we have the ring 0 covers sections 1, 2, 3, 4, the ring 1 covers sections 1, 2, 3, 4, and ring 2 covers sections 6, 7, 8, 9. The sections 0, 5, 10, 11 are not blocked by any arc.
After the third rotation (which is rotating ring 1 clockwise by 1 section), we have the ring 0 covers sections 1, 2, 3, 4, the ring 1 covers sections 2, 3, 4, 5, and ring 2 covers sections 7, 8, 9, 10. The sections 0, 6, 11 are not blocked by any arc.
If we rotate ring 0 clockwise once, we can see that the sections ring 0 covers will be the same as the sections that ring 1 covers, hence p_1=1.
If we rotate ring 0 clockwise five times, the sections ring 0 covers will be the same as the sections that ring 2 covers, hence p_2=5.
Note that if we will make a different set of rotations, we can end up with different values of p_1 and p_2 at the end.