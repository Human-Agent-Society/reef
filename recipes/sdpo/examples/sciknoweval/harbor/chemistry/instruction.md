# Learn SciKnowEval Chemistry through Reef

This task trains the model Reef serves with the `sdpo` recipe
(arXiv:2601.20802) on the Chemistry split of SciKnowEval, the generalization
sweep of the reference implementation
([lasgroup/SDPO](https://github.com/lasgroup/SDPO) at `7c457fc1b1f6`). A prompt
is the authors' system message fixing the `<reasoning>`/`<answer>` format and a
four-option chemistry question; an answer is correct when the letter between
its last `<answer>` tags matches the dataset's.

The harness runs `python /opt/chemistry/stage.py` in this container. For each
step the runner samples 32 questions 8 times each through the Reef service at
`$REEF_SERVICE_URL`, reports every rollout against its receipt with its
coordinates in the grid and its score, and waits for the step's training
release before sampling the next grid. The recipe holds the step until the grid
is complete, then each rollout's teacher rereads the question with a successful
sibling's response, and its distribution over the student's own tokens is the
target of one optimizer step.

Every five steps, and before the first, the runner evaluates avg@16 on the 210
test questions at temperature 0.6 and top-p 0.95. That series is the learning
curve, and the task's reward is its last value.
