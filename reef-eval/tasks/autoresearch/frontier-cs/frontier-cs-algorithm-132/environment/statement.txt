time limit per test: 0.25 second
memory limit per test: 512 megabytes
This is an interactive problem.
You have 𝑅 vacuum cleaner robots left over from your trade with your fellow contestants. You want to use these robots to locate the two chairmen. You can instruct each robot to scout several of 1000 positions where the chairmen could be located.
Each robot can only detect whether or not there is at least one chairman at its scouted positions. Every robot needs a full hour to scout its positions before returning with its result to you. Because this will drain the robot’s battery, you can send out each robot only once.
You want to know the positions of the chairmen after at most 𝐻 hours. In particular, you might be forced to send out several robots at once without waiting for the previous robots to return. You can assume that the two chairmen stay at the same positions all the time.
Write a program which plans this scouting mission and determines where the two chairmen are located.


Input
The first line consists of 2 integers R and H (R=75,H=1), indicating the number of vacuum cleaner robots and the number of hours.

Interaction
To send a robot to scout the positions, print on a single line "? k, 𝑃[0],…,𝑃[𝑘 − 1]" (where 𝑘 is the length of the array 𝑃). The positions 𝑃[𝑖] must be pairwise distinct integers between 1 and 1000. You can 'send' at most 𝑅 times per testcase.
To get the answers of robot that sent before, print on a single line '@', then you will receive an integer L and a array (with length L) with exactly one entry for each robot sent out one hour ago (by a 'send' after the previous 'get' to wait or after the beginning of the program). The entry at index 𝑖 is 1 if the (𝑖 +1)-th of these robots has detected at least one chairman at its scouted positions, and 0 otherwise. You can 'get' at most 𝐻 times per testcase.
Once you have figured out the positions, print '!' followed by 2 integers a and b (1 ≤ 𝑎,𝑏 ≤ 1000, 𝑎 = 𝑏 is allowed), representing the positions of the two chairmen.
After printing a query do not forget to output the end of line and flush the output. Use fflush(stdout) or cout.flush() to flush.

Grading
your actual score depends on the number rmax of robots sent out. Specifically,
rmax<=30, score=-20/3*rmax+820/3;
30<rmax<=35, score=-4*rmax+580/3;
35<rmax<=40, score=-8/3*rmax+440/3;
40<rmax<=60, score=-4/3*rmax+280/3;
60<rmax<=75, score=13;
75<rmax, score=0;

Example
Consider a testcase with 𝑅 = 75 and 𝐻 = 20 where the chairmen are located at positions 13 and 37.
First, the grader calls your function scout as scout(75,20). Then, an interaction between your program and the grader could look as follows:
Your program | read | Explanation
? 3 42 13 37 | none | send a robot to positions 13, 37 and 42
? 2 47 11    | none |send a robot to positions 13, 37 and 42
@            | 2 1 0|wait an hour for the return of two robots; only the first robot has detected a chairman
? 1 42       | none |send a robot to position 42
@            | 1 0  |wait an hour; there is no chairman at position 42 you are convinced that the chairmen are located
! 13 37      | none |you are convinced that the chairmen are located at positions 13 and 37

Returning the pair {37,13} would be accepted as well.
Note that the above queries are of course not sufficient to determine the positions of the chairmen with certainty: For example, both being at position 37 or one chairman being at position 13 while the other is at position 100 would also be consistent with all answers to wait, so the grader could also have rejected this solution.
Moreover, the grader is adaptive, i.e. the positions of the chairmen may depend on the behavior of your program in the current as well as in earlier runs.