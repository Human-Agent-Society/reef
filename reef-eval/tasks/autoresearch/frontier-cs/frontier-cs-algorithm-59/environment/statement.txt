# Limited Shuffle Restoring

**Input file:** standard input  
**Output file:** standard output  
**Time limit:** 3 seconds  
**Memory limit:** 512 meblbytes  

**This is an interactive problem.**

Bobo had an array \\( a \\), initially equal to \\((1,2,\ldots,n)\\). He did the following operations with the array.

- For each \\( i \\) from 1 to \\( n \\) in this order, Bobo picked some index \\( j \\) such that \\( i \leq j \leq \min(n,i+2) \\), and swapped \\( a_i \\) and \\( a_j \\). Of course, if \\( i = j \\), then nothing happened after the operation.

Your goal is to determine the final array. You may ask questions of the following type.

- ? i j meaning the question "How do \\( a_i \\) and \\( a_j \\) compare to each other?". Bobo will respond to this with one symbol < or >, meaning that \\( a_i < a_j \\) or \\( a_i > a_j \\), respectively.

You may ask no more than \\(\lfloor 5n/3 \rfloor + 5\\) questions. After this, you must guess the array.

## Interaction Protocol

First, the interactor prints the number \\( n \\) in a separate line (\\( 1 \leq n \leq 30,000 \\)). Then the solution makes queries, where each query consists of printing ? i j on a separate line, where \\( 1 \leq i,j \leq n \\), and \\( i \neq j \\). After each query the interactor prints one character < or > on a separate line.

After the solution has finished asking questions, it must make a guess. If you think that the array is \\((a_1,\ldots,a_n)\\), print ! \\( a_1 \ a_2 \ldots \ a_n \\) on a separate line and terminate.

If your solution makes more than \\(\lfloor 5n/3 \rfloor + 5\\) queries, the interactor will finish with the WA verdict. If you do not flush the output after printing a query, you may receive the IL verdict.

Note that the interactor in this task is **adaptive**, i.e. the array may be generated at the runtime consistently with your questions.

## Example

| standard input | standard output |
|----------------|-----------------|
| 5              | ? 5 4           |
| <              | ? 5 1           |
| >              | ? 5 3           |
| >              | ? 3 1           |
| <              | ? 2 1           |
| >              | ? 5 2           |
| >              | ! 2 3 1 5 4     |