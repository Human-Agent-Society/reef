Time Limit: 10s
Memory Limit: 1024M

Firstly, you are given two integers n (1 <= n <= 400) and m (1 <= m <= 4000), which means that you have n elements and m sets.

After that, there are m integers, the i-th integer is the cost of choosing the i-th set.

After that, for the i-th element, firstly input an integer k_i, which means the number of sets that contain the element. After that, there

are k_i integers, the j-th integer a_j means that the set with id a_j contains the element i.

Find some sets so that each element belongs to at least one of these sets. You need to minimize the total cost of these sets. This value will determine your final score.

Output: 

Firstly output an integer |S|, which means the number of sets you choose. After that, output |S| ids of the sets you choose in another line.

