# Train through Reef on an arithmetic grid

This task trains the model Reef serves with the `sdpo` recipe
(arXiv:2601.20802) on a synthetic arithmetic grid: each question asks for the
sum of two integers and requires the answer as a bare integer, so a wrong
answer is a formatting or arithmetic failure the environment can describe.

The harness runs `python /opt/grid/grid.py` in this container. The runner
samples every question of the step's grid `SDPO_ROLLOUTS_PER_GROUP` times
through the Reef service at `$REEF_SERVICE_URL`, reports each rollout against
its receipt with its coordinates in the grid and its score, and waits for the
step's training release before sampling the next grid.

The recipe holds the step until every coordinate has arrived, then builds the
whole grid as one batch: each rollout's teacher reads the original question
plus the first successful sibling's response, or the environment's feedback
when the question had no success. The task's reward is the last grid's
accuracy.
