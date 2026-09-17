# Learn Science Q&A through Reef

This stage trains the model Reef serves on the Science Q&A skill of the
reference implementation of Self-Distillation Fine-Tuning
(arXiv:2601.19897): the Chemistry L-3 subset of SciKnowEval, where a prompt
is a system message fixing the `<reasoning>`/`<answer>` format and a
four-option chemistry question, and the demonstration is GPT-4o's response.

The harness runs `python /opt/skills/stage.py` in this container. The runner
samples each step's prompts through the Reef service at `$REEF_SERVICE_URL`,
reports every demonstration against the sample's receipt, and waits for the
step's training release. Every `SKILLS_EVAL_EVERY` steps, and after the last,
it submits the step number to the judge at `$JUDGE_URL`, which scores the
served model on the test splits of both skills (Science Q&A: exact match of
the answer letter; Tool Use: regex match of the API call) and records the
scores.

The stage's reward is the Science Q&A accuracy after the last step; the Tool
Use accuracy rides along on every submission, which is where forgetting
shows.
