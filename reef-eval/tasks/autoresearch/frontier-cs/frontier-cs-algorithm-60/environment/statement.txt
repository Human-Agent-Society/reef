```markdown
# Problem K: Probing the Disk

**Time Limit: 2 seconds**

*This is an interactive problem.*

A thin black disk is laid flat on the square bottom of a white box. The sides of the box bottom are 10^5 units long.

Somehow, you are not allowed to look into the box, but you want to know how large the disk is and where in the box bottom the disk is laid. You know that the shape of the disk is a true circle with an integer units of radius, not less than 100 units, and its center is integer units distant from the sides of the box bottom. The radius of the disk is, of course, not greater than the distances of the center of the disk from any of the sides of the box bottom.

You can probe the disk by projecting a thin line segment of light to the box bottom. As the reflection coefficients of the disk and the box bottom are quite different, from the overall reflection intensity, you can tell the length of the part of the segment that lit the disk.

Your task is to decide the exact position and size of the disk through repetitive probes.

## Interaction

You can repeat probes, each of which is a pair of sending a query and receiving the response to it. You can probe at most 1024 times.

A query should be sent to the standard output in the following format, followed by a newline.

```
query x1 y1 x2 y2
```

Here, (x1, y1) and (x2, y2) are the positions of the two ends of the line segment of the light. They have to indicate distinct points. The coordinate system is such that one of the corners of the box bottom is the origin (0, 0) and the diagonal corner has the coordinates (10^5, 10^5). All of x1, y1, x2, and y2 should be integers between 0 and 10^5, inclusive.

In response to this query, a real number is sent back to the standard input, followed by a newline. The number indicates the length of the part of the segment that lit the disk. It is in decimal notation without exponent part, with 7 digits after the decimal point. The number may contain an absolute error up to 10^−6.

When you become sure about the position and the size of the disk through the probes, you can send your answer. The answer should have the center position and the radius of the disk. It should be sent to the standard output in the following format, followed by a newline.

```
answer x y r
```

Here, (x, y) should be the position of the center of the disk, and r the radius of the disk. All of x, y, and r should be integers.

After sending the answer, your program should terminate without any extra output. Thus, you can send the answer only once.

### Notes on interactive judging

When your output violates any of the conditions above (incorrect answer, invalid format, x1, y1, x2, or y2 being out of the range, too many queries, any extra output after sending your answer, and so on), your submission will be judged as a wrong answer. As some environments require flushing the output buffers, make sure that your outputs are actually sent. Otherwise, your outputs will never reach the judge.

You are provided with a command-line tool for local testing. For more details, refer to the clarification in the contest system.

**Figure K.1. Sample Interaction**

Read

```
60000.0000000
0.0000000
12315.3774869
```

Write
```
query 40000 0 40000 100000
query 0 10000 100000 10000
query 60000 60000 80000 80000
answer 40000 60000 30000
```