# Learn Tool Use through Reef

This stage trains the model Reef serves on the Tool Use skill of the
reference implementation of Self-Distillation Fine-Tuning
(arXiv:2601.19897): ToolAlpaca's training split, where a prompt is a
tool's documentation and a user request in the ReAct format and the
demonstration is the dataset's golden response.

The harness runs `python /opt/skills/stage.py` in this container. The runner samples each step's prompts through the
Reef service at `$REEF_SERVICE_URL`, reports every demonstration against the
sample's receipt, and waits for the step's training release. Every
`SDFT_EVAL_EVERY` steps, and after the last, it submits the step number to
the judge at `$JUDGE_URL`, which scores the served model on the test splits
of both skills (Tool Use: regex match of the API call; Science Q&A: exact
match of the answer letter) and records the scores.

The stage's reward is the Tool Use accuracy after the last step; the Science
Q&A accuracy rides along on every submission.
