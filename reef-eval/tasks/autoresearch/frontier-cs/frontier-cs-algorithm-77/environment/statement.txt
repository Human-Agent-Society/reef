Problem:
Time limit: 3 seconds
Memory limit: 512 megabytes
The popular improv website Interpretation Impetus hosts regular improv contests and maintains a rating
of the best performers. However, since improv can often go horribly wrong, the website is notorious for
declaring improv contests unrated. It now holds a wager before each improv contest where the participants
try to predict whether it will be rated or unrated, and they are now more popular than the improv itself.
Izzy andnother participants take part in each wager. First, they each make their prediction, expressed as
1 (“rated”) or0 (“unrated”). Izzy always goes last, so she knows the predictions of the other participants
when making her own. Then, the actual competition takes place and it is declared either rated or unrated.
You need to write a program that will interactively play as Izzy. There will be m wagers held in 2021,
and Izzy’s goal is to minimize the number of wrong predictions after all those wagers. Izzy knows nothing about the other participants /emdash.cyr they might
somehow always guess correctly, or their predictions might be correlated. Izzy’s predictions, though, do
not aﬀect the predictions of the other participants and the decision on the contest being rated or not /emdash.cyr in
other words, in each test case, your program always receives the same inputs, no matter what it outputs.
Interaction Protocol
First, a solution must read two integers n (1 ≤n ≤1000) and m (1 ≤m ≤10 000). Then, the solution
must process m wagers. For each of them, the solution must ﬁrst read a string consisting of n 0s and 1s,
in which the i-th character denotes the guess of the i-th participant. Then, the solution must print Izzy’s
guess as 0 or 1. Don’t forget to ﬂush the output after printing it! Then, the solution must read the actual
outcome, also as 0 or 1, and then proceed to the next wager, if this wasn’t the last one.
Suppose your solution has c wrong predictions, your score will be min((2*b-c)/b, 1)*100, where b is the smallest
number of mistakes made by any other participant. If you . Note that if a solution outputs anything except 0 or
1 for a wager, it will be considered incorrect even if it made no other mistakes.
There are 200 test cases in this problem.
Example
standard input standard output
3 4
000
1
100
1
001
0
111
1
0
0
1
1
Note
In the example, the participants made 1, 2, and 3 mistakes respectively, thereforeb= 1(the smallest of
these numbers). Izzy made 3 mistakes, which were not more than 1.3 ·b + 100 = 101.3, so these outputs
are good enough to pass this test case (as are any other valid outputs).
Page 12 of 14