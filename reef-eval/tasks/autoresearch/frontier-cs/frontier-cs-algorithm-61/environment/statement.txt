# Let's Go! New Adventure

**Input file:** standard input  
**Output file:** standard output  
**Time limit:** 3 seconds  
**Memory limit:** 1024 megabytes

In Pigeland, Pishin is a popular open-world action RPG where users can play multiple characters. Each character has an independent adventure rank, which increases as they earn experience points (EXP) while being played. Initially, every character starts with an adventure rank of level 0 and can progress up to a maximum level of \( m \). To advance from level \((i-1)\) to level \( i \) (\( 1 \leq i \leq m \)), the character is required to earn \( b_i \) EXP.

Grammy plans to play Pishin for the next \( n \) days. As a rich girl, her Pishin account has an infinite number of characters. However, being a lazy girl, all characters in her account start with an adventure rank of level 0 at the beginning of the \( n \) days. Each day, Grammy will select exactly one character to play, but once she stops playing a character, she cannot resume playing that character on any future day. In other words, she can only continue playing the same character on consecutive days.

On the \( i \)-th day, Grammy will earn \( a_i \) EXP for the character she plays. This means that if she plays a character continuously from the \( l \)-th day to the \( r \)-th day (both inclusive), the character's adventure rank will increase to level \( k \), where \( k \) is the largest integer between 0 and \( m \) such that the total EXP earned (which is \(\sum_{i=l}^{r} a_i\)) is greater than or equal to the requirement of leveling up to \( k \) (which is \(\sum_{i=1}^{k} b_i\)).

Being a greedy girl, Grammy wants to maximize the total sum of adventure ranks across all her characters after the \( n \) days. However, as a single-minded girl, she doesn't want to play too many different characters. To balance this, she introduces a penalty factor of \( c \). Her goal is to maximize the total sum of adventure ranks across all characters after the \( n \) days, minus \( c \times d \), where \( d \) is the number of different characters she plays. As Grammy's best friend, your task is to compute the maximum value she can achieve under the optimal strategy for selecting characters.

## Input

There are multiple test cases. The first line of the input contains an integer \( T \) (\( 1 \leq T \leq 5 \times 10^4 \)) indicating the number of test cases. For each test case:

- The first line contains three integers \( n \), \( m \) and \( c \) (\( 1 \leq n, m \leq 5 \times 10^5, 0 \leq c \leq 5 \times 10^5 \)).
- The second line contains \( n \) integers \( a_1, a_2, \cdots, a_n \) (\( 0 \leq a_i \leq 10^{12}, 0 \leq \sum_{i=1}^{n} a_i \leq 10^{12} \)).
- The third line contains \( m \) integers \( b_1, b_2, \cdots, b_m \) (\( 0 \leq b_i \leq 10^{12}, 0 \leq \sum_{i=1}^{m} b_i \leq 10^{12} \)).

It is guaranteed that neither the sum of \( n \) nor the sum of \( m \) of all test cases will exceed \( 5 \times 10^5 \).

## Output

For each test case, output one line containing one integer, indicating the maximum value.

## Example

**standard input**
```
2
5 4 2
1 0 3 1 2
0 1 1 2
4 5 1
7 16 23 4
1 3 6 20 20
```

**standard output**
```
3
6
```

## Note

For the first sample test case, one solution is to use the first three days to get a character with adventure rank 4 and the next two days to get another character with adventure rank 3. This gives us a value of \((4-2)+(3-2)=3\).

For the second sample test case, we can play a different character each day; this gives us adventure ranks 2, 3, 3, and 2, respectively. So the value is \((2-1)+(3-1)+(3-1)+(2-1)=6\).