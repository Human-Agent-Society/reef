Problem: Texas Hold’em Training (Terminal I/O Interactive)

Time limit: 10 seconds
Memory limit: 512 MB

Overview
You will write a program that plays a large number of very simplified heads-up Texas Hold’em hands against a fixed opponent policy. The judge runs the game and reveals exactly the information you are allowed to know. Your program must decide whether to CHECK, FOLD, or RAISE an integer number of chips each betting round to maximize your chip profit. Interaction is via standard input/output (stdin/stdout). You must flush after every line you print.

Cards, ranks, and hand comparison
- Deck: 52 cards, 4 suits labeled 0,1,2,3. Each suit has 13 values labeled 1..13 corresponding to 2,3,4,5,6,7,8,9,T,J,Q,K,A (in strictly increasing order).
- A 5-card hand type ranking (highest to lowest):
  1) Straight flush (a straight with all five cards same suit)
  2) Four of a kind
  3) Full house (three of a kind + a pair)
  4) Flush (five cards same suit)
  5) Straight (five consecutive values; A-2-3-4-5 is valid and is the lowest straight; K-A-2-3-4 is not)
  6) Three of a kind
  7) Two pairs
  8) One pair
  9) High card
- Comparing two hands:
  - If hand types differ, higher type wins.
  - If both are straights or straight flushes: compare by the straight’s rank. Ten-to-Ace (T-J-Q-K-A) is the highest; A-2-3-4-5 is the lowest.
  - Otherwise, sort the 5 cards by the tuple (multiplicity, value) in descending order, where multiplicity is how many times the value appears in the hand. Compare these 5-card sequences lexicographically. Suits never break ties beyond determining flush/straight flush.
  - Examples:
    - Same-suit 5-6-7-8-9 > same-suit A-2-3-4-5.
    - Same-suit A-2-3-4-5 > same-suit 2-4-5-6-7 (the latter is not a straight).
    - 3-3-8-8-K > 5-5-7-7-A (two pairs with higher top pair/tiebreakers).
    - Q-Q-Q-T-T > J-J-J-A-A.
- Board-of-7 rule: At showdown each player forms their best 5-card hand from their 7 available cards (2 private + 5 community). The higher best 5-card hand wins; equal best hands tie.

Game flow (per hand)
- Shuffling and dealing:
  - The 52 cards are uniformly randomly permuted.
  - You (Alice) receive the top 2 cards; the opponent (Bob) receives the next 2 (unknown to you).
  - A “pot” starts with 10 chips. Both players start the hand with 100 chips behind (stacks). All chip counts are integers.
- Four betting rounds, with at most one action from you and one response from Bob per round:
  1) Round 1 (preflop): no community cards are visible.
  2) Reveal 3 community cards (the flop).
  3) Round 2.
  4) Reveal 1 community card (the turn).
  5) Round 3.
  6) Reveal 1 community card (the river).
  7) Round 4.
- Your options when it is your turn in any round:
  - CHECK: no chips move.
  - FOLD: the hand ends immediately; Bob wins the entire pot.
  - RAISE x: choose an integer x with 1 ≤ x ≤ your current stack; you move x chips into the pot.
- Bob’s response after your action:
  - If you CHECK: Bob always CHECKS (no bet this round).
  - If you RAISE x: Bob either FOLDs (you win the pot immediately) or CALLs (she moves x chips into the pot).
  - It is guaranteed Bob always has enough chips to call any allowed raise x (by construction of this single-raise-per-round process).
- Showdown and payouts:
  - If nobody folds by the end of Round 4, reveal all community cards (already visible) and both hole cards are implicitly known to the judge; the judge determines the winner using the rules above.
  - Winner gets the entire pot; if tie, the pot is split evenly (integer arithmetic; pot is always even here).
  - Your profit for the hand is (your ending stack) − 100. Positive means you won chips; negative means you lost chips.

Opponent policy (Bob)
- If you CHECK, she CHECKs.
- If you RAISE x, she compares:
  - EV(FOLD) = (her current stack) − 100.
  - EV(CALL) = estimated via 100 Monte Carlo rollouts:
    - Consider all yet-unseen cards (your hole cards are hidden from Bob, and unrevealed community cards are unknown).
    - For each rollout: take a uniform random permutation of the remaining unseen cards; assign them to all unknown positions in natural order (your hidden cards remain unknown to Bob but are assigned in the simulation; remaining community cards are dealt next).
    - Assume that after she calls now, you will CHECK in all future rounds (this is how she evaluates the call).
    - Compute her resulting profit in that simulation.
    - EV(CALL) is the average over the 100 rollouts.
  - She CALLs if and only if EV(CALL) > EV(FOLD); otherwise she FOLDs.

Terminal I/O protocol
All lines are plain ASCII tokens separated by spaces. You must flush after every line you print. If you ever read -1, you must exit immediately.

Start of the match
- Read a single integer G: the number of hands the judge will play (up to 10,000 in official tests).

Per-hand loop
The judge will drive the hand by repeatedly sending a STATE describing your next decision point. After a STATE, you may ask for Monte Carlo equity estimates, then you must output exactly one ACTION.

State description (from judge to you)
- STATE h r a b P k
  - h: 1-based hand index
  - r: current round in {1,2,3,4} (you act first in each round)
  - a: your current stack (chips behind)
  - b: Bob’s current stack
  - P: current pot size
  - k: number of currently revealed community cards; k ∈ {0,3,4,5}
- ALICE c1 v1 c2 v2
  - Your two hole cards as (suit, value) pairs; suit in [0..3], value in [1..13] for 2..A
- BOARD followed by 2k integers
  - Exactly k cards, each as (suit, value); if k = 0, the line is just “BOARD” with no numbers.

Optional helper query (from you to judge)
- RATE t
  - t: positive integer number of sampling rollouts the judge should use to estimate your win/draw rates from the current partial state (this mimics getRatesBySampling). The judge responds:
  - RATES w d
    - w: estimated probability that your final 7-card hand will be strictly better than Bob’s (double)
    - d: estimated probability of a tie (double)
  - The sampling completes the currently unknown cards (Bob’s hole, unrevealed community) uniformly at random from the remaining deck.
  - Global budget: sum of all t over the entire match must be ≤ 3,000,000. If you exceed this, the judge may reply with -1 and terminate. RATE does not advance the hand; you may issue multiple RATE queries per STATE within the budget.

Your decision (exactly one per STATE)
- ACTION CHECK
- ACTION FOLD
- ACTION RAISE x
  - x must be an integer with 1 ≤ x ≤ your current stack a.

Judge’s immediate response after your ACTION
- If ACTION CHECK:
  - OPP CHECK
  - If r < 4, dealing proceeds and you will receive the next STATE for round r+1 (with updated k = 3,4,5).
  - If r = 4, the hand ends by showdown; judge then prints RESULT delta.
- If ACTION RAISE x:
  - OPP FOLD
    - The hand ends immediately; judge prints RESULT delta (your profit for this hand).
  - or OPP CALL x
    - The hand continues. If r < 4, the judge proceeds to the next STATE (round r+1 with more community cards). If r = 4, the hand ends by showdown and judge prints RESULT delta.
- RESULT delta
  - delta is your integer profit for this hand: ending stack − 100. The next hand then begins (or the match ends if h = G).

End of the match
- After all G hands, the judge prints:
  - SCORE W
    - W is your average profit per hand (double), i.e., mean of all delta.

Validity and termination
- Any malformed command, out-of-range raise, or protocol violation may cause the judge to print -1 and terminate immediately; your program must exit upon reading -1.
- Always flush after printing RATE or ACTION (e.g., in C++: cout << line << endl; or fflush(stdout)).

Constraints and guarantees
- G ≤ 10,000 in official tests.
- Sum of t over all RATE queries ≤ 3,000,000.
- All chip movements are integers. Pot and stacks fit in 32-bit signed integers in all official tests.
- The hidden deck for each hand is fixed before the hand begins and does not depend on your queries.
- With this single-raise-per-round structure and symmetric stacks, Bob always has enough chips to CALL any legal RAISE you declare.

Scoring
Only programs that follow the protocol, do not exceed the RATE budget, and finish all hands are scored.

Let W be the final average profit per hand printed by the judge (the SCORE value). Your points are a piecewise-linear function of W:

- If W ≤ 8.0: score = 0.
- If 8.0 < W ≤ 11.0: score increases linearly from 0 to 40:
  score = round(13.3 × (W − 8)).
- If 11.0 < W ≤ 14.0: score increases linearly from 40 to 82:
  score = 40 + round(14 × (W − 11)).
- If 14.0 < W: score increases linearly from 82 to Infinite:
  score = 82 + round(3 × (W − 14)).


Interpretation:
- Around W ≥ 11 indicates you beat a simple baseline (roughly “Small Task”).
- Around W ≥ 16 indicates a strong strategy (roughly “Large Task”).

Example interaction (illustrative only)
Judge: 1
Judge: STATE 1 1 100 100 10 0
Judge: ALICE 0 12 2 1
Judge: BOARD
You: RATE 100
Judge: RATES 0.421000 0.010000
You: ACTION RAISE 5
Judge: OPP FOLD
Judge: RESULT 15
Judge: SCORE 15.000000

Notes and clarifications
- Card encoding:
  - Each card is printed as two integers: suit in [0..3], value in [1..13] corresponding to 2..A.
  - Community cards appear in deal order on BOARD (first the 3-card flop, then turn, then river).
- You do not need to implement hand evaluation to be correct; the judge handles showdowns. However, to plan your actions you may use RATE queries (within budget) or implement your own simulation/evaluation.
- Precision of RATES and SCORE is implementation-defined by the judge; treat them as doubles. Do not rely on a fixed number of decimals.
- If you ever read -1, exit immediately without printing anything further.

Strategy discussion (non-binding)
- The opponent’s CALL decision underestimates your future aggression (she assumes you will CHECK afterwards), which can be exploited by well-timed raises.
- RATE queries give you (win, tie) probabilities under random completions; combined with current pot and effective stacks, you can estimate immediate fold equity versus call equity to choose raise sizes.
- Budget your RATE calls: preflop and early-street coarse estimates (small t) and larger t near pivotal decisions can perform well within the 3,000,000 budget.
