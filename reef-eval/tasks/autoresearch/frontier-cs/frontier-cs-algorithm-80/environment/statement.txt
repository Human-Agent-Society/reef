Indiana Jones and the Uniform Cave

This is an I/O interactive problem. I/O interaction refers to interactive problems, where the program communicates with a special judge during execution instead of producing all output at once. In these problems, the program sends queries (output) to the judge and must immediately read responses (input) before continuing. The solution must strictly follow the input-output protocol defined in the problem statement, because any extra output, missing flush, or incorrect format can cause a wrong answer. Unlike standard problems, interactive problems require careful handling of I/O, synchronization, and flushing to ensure smooth communication between the contestant’s code and the judge.

Indiana Jones has stuck in the Uniform Cave.
There are many round chambers in the cave, and all of them are indistinguishable from each other.
Each chamber has the same number of one-way passages evenly distributed along the chamber’s wall.
Passages are indistinguishable from each other, too.

The Cave is magical. All passages lead to other chambers or to the same one.
However, the last passage, after all passages are visited, leads to the treasure.
Even the exact number of chambers is a mystery.
It is known that each chamber is reachable from each other chamber using the passages.

Dr. Jones noticed that each chamber has a stone in the center.
He decided to use these stones to mark chambers and passages.
A stone can be placed to the left or to the right of one of the passages.
When Indiana Jones enters the chamber all that he can observe is the location of the stone in the chamber.
He can move the stone to the desired location and take any passage leading out of the chamber.

Your task is to help Indiana Jones to visit every passage in the Uniform Cave and find the treasure.

Interaction Protocol

First, the testing system writes the integer m — the number of passages in each chamber (2 ≤ m ≤ 20).

Dr. Jones enters the chamber and sees, in the next line, where the stone is placed:
either in the “center” of the chamber or to the “left”, or to the “right” of some passage.
On the first visit to the chamber, the stone is in the center.

Your solution shall output his actions:
the number and the side of the passage to place the stone to,
and the number of the passage to take.
Both numbers are relative to the passage marked by the stone, counting clockwise from 0 to m−1.
If the stone is in the center of the chamber, the origin is random.

For example,
3 left 1
tells that Dr. Jones moves the stone three passages clockwise and places it to the left of the passage,
then he takes the passage to the right of the initial stone position.

After each move, the testing system tells either the location of the stone in the next chamber
or “treasure”, if Indiana Jones had found it.
The testing system writes “treasure” when all the passages are visited.

If Dr. Jones does not find the treasure room after 50 000 passages are taken, he starves to death,
and your solution receives the “Wrong Answer” outcome.
You also receive this outcome if your solution terminates before all passages are taken. Let q be the number of passages you have taken, your score will be (50000 - q) / 50000.

The total number of chambers in the cave is unknown,
but you may assume that it does not exceed 20,
and that each chamber is reachable from every other chamber.

Example

Standard Input
2
center
left
center
left
right
treasure

Standart Output
0 left 0
1 left 1
1 right 0
0 left 0
1 right 0

Dr. Jones enters the example cave and sees that the stone in the first chamber is in the center.
He marks the chamber by placing the stone to the left of some passage and takes it.
He sees the chamber where the stone is to the left of the passage, so he is in the first chamber again.
He moves the stone clockwise and takes the passage marked by it.
This passage leads to the second chamber.
He marks it by placing the stone to the right of some passage and takes another one.
He is in the first chamber again, so he returns to the second chamber and takes the remaining passage.
This passage leads to the treasure.