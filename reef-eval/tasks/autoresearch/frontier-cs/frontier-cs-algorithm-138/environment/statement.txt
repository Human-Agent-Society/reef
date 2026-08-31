### Problem C. Fabulous Fungus Frenzy

As the Traveler in the game Genshin Impact, you are exploring Sumeru. You are invited to the Nilotpala Cup Beast Tamers Tournament. To win, you must pass the Coruscating Potential challenge to cultivate Fungi.

During this challenge, you must use Floral Jellies to form blends for your fungi. You are given an initial configuration of jellies, represented by an n x m matrix. Each entry (i, j) contains a Floral Jelly. The value of the entry represents its type, and equal values mean they are the same type.

The goal is to turn the initial configuration into a target configuration. After all operations, each position must be occupied by a Floral Jelly of the required type.

You can use three kinds of operations: Switch, Rotate, and Preset.

  * **Switch:** Exchange the positions of any two adjacent Floral Jellies. Two jellies at (x1, y1) and (x2, y2) are adjacent if |x1 - x2| + |y1 - y2| = 1.
  * **Rotate:** Select any 2x2 block of jellies at (x, y), (x, y+1), (x+1, y+1), and (x+1, y). Shift their positions one step in a clockwise direction. The new jellies at these positions will be the ones previously at (x+1, y), (x, y), (x, y+1), and (x+1, y+1), respectively.
  * **Preset:** Choose a pre-existing n' x m' formula (Matrix F) and a top-left starting position (x, y). Replace all jellies in the block from (x, y) to (x+n'-1, y+m'-1) with the jellies from the formula F.

-----

### Input

The first line contains three integers n, m, and k (2 \<= n, m \<= 20, 1 \<= k \<= 20), indicating the size of the jelly configuration and the number of preset formulas.

The following n lines each contain a string of m characters, representing the initial n x m configuration.

An empty line follows.

The following n lines each contain a string of m characters, representing the target n x m configuration.

Then, k preset formulas follow. Each preset formula starts with an empty line. The next line contains two integers np and mp (1 \<= np \<= n, 1 \<= mp \<= m), indicating the matrix size of the preset. The following np lines contain the mp-character strings for that formula.

There are 62 types of Floral Jellies, denoted by 'a'-'z', 'A'-'Z', and '0'-'9'.

-----

### Output

If the puzzle is unsolvable, output "-1".

Otherwise, output an integer r (0 \<= r \<= 4x10^5) in the first line, indicating the number of moves needed.

Then, output r lines, each containing three integers op, x, and y, describing an operation. The jelly at (x, y) is in the x-th row from the top and y-th column from the left. The operations are:

  * **-4 x y**: Swaps jellies at (x, y) and (x+1, y). Requires 1 \<= x \< n, 1 \<= y \<= m.
  * **-3 x y**: Swaps jellies at (x, y) and (x-1, y). Requires 1 \< x \<= n, 1 \<= y \<= m.
  * **-2 x y**: Swaps jellies at (x, y) and (x, y-1). Requires 1 \<= x \<= n, 1 \< y \<= m.
  * **-1 x y**: Swaps jellies at (x, y) and (x, y+1). Requires 1 \<= x \<= n, 1 \<= y \< m.
  * **0 x y**: Rotates the 2x2 block at (x, y), (x, y+1), (x+1, y+1), and (x+1, y) clockwise. Requires 1 \<= x \< n, 1 \<= y \< m.
  * **op x y** (where 1 \<= op \<= k): Covers the submatrix starting at (x, y) with the op-th preset formula. Requires 1 \<= x \<= n-nop+1 and 1 \<= y \<= m-mop+1.

The total number of preset operations (op \>= 1) cannot exceed 400. The total number of operations cannot exceed 4x10^5. You do not need to minimize the number of operations.

-----

### Examples

**Example 1**

**Input:**

```
3 3 1
000
GOG
BGB

000
GGG
BBB

3 1
B
G
B
```

**Output:**

```
4
1 1 3
0 1 2
-1 3 2
-4 3 3
```

**Example 2**

**Input:**

```
2 2 1
00
00

PP
PP

1 2
OP
```

**Output:**

```
-1
```

**Example 3**

**Input:**

```
4 8 4
11122222
33344444
55556666
77777777

NIXSHUOX
DEXDUIxx
DANXSHIX
YUANSHEN

2 3
NIy
DEX

3 8
ZZZZZZZZ
DANNSH9I
YUA9SHEN

1 1
X

2 5
SH08y
DUUI8
```

**Output:**

```
13
2 2 1
-3 3 4
-2 3 8
1 1 1
4 1 4
0 1 6
3 1 3
3 1 8
3 2 3
3 2 7
3 2 8
3 3 4
3 3 8
```