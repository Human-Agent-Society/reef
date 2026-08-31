Playing Around the Table

## Problem Description

There are n players, numbered from 1 to n sitting around a round table. The (i+1)-th player sits to the right of the i-th player for 1≤i<n, and the 1-st player sits to the right of the n-th player.

There are n^2 cards, each of which has an integer between 1 and n written on it. For each integer from 1 to n, there are exactly n cards having this number.

Initially, all these cards are distributed among all the players, in such a way that each of them has exactly n cards. In one operation, each player chooses one of his cards and passes it to the player to his right. All these actions are performed simultaneously.

Player i is called solid if all his cards have the integer i written on them. Their objective is to reach a configuration in which everyone is solid. Find a way to do it using at most (n^2−n) operations.

## Input Format

The first line contains a single integer n (2≤n≤100).

Then n lines follow. The i-th of them contains n integers c1,c2,…,cn (1≤cj≤n) — the initial cards of the i-th player.

It is guaranteed that for each integer i from 1 to n, there are exactly n cards having the number i.

## Output Format

In the first line print an integer k (0≤k≤(n^2−n)) — the numbers of operations you want to make.

Then k lines should follow. In the i-th of them print n integers d1,d2,…,dn (1≤dj≤n) where dj is the number written on the card which j-th player passes to the player to his right in the i-th operation.

We can show that an answer always exists under the given constraints. If there are multiple answers, print any.

## Example

### Input

```
2
2 1
1 2
```

### Output

```
1
2 1
```

### Input

```
3
1 1 1
2 2 2
3 3 3
```

### Output

```
6
1 2 3
3 1 2
2 3 1
1 2 3
3 1 2
2 3 1
```

## Note

In the first test case, if the first player passes a card with number 2 and the second player passes a card with number 1, then the first player has two cards with number 1 and the second player has two cards with number 2. Then, after making this operation, both players are solid.

In the second test case, 0 operations would be enough too. Note that you do not need to minimize the number of operations.

## Constraints

- 2 ≤ n ≤ 100
- For each integer i from 1 to n, there are exactly n cards having the number i

## Time Limit

2 seconds per test case

## Memory Limit

256 MB