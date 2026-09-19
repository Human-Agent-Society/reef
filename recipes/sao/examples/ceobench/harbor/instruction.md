Run one CEO-Bench episode: operate the NovaMind AI startup for the configured
number of simulated days, starting from $1,000,000 in cash, through the
benchmark's bash agent. The episode's system prompt, tools, and simulator come
from the pinned CEO-Bench checkout at `/opt/ceobench`; this instruction is not
shown to the model.

The run directory lands under `/workspace/ceobench-runs/run_<id>/`. The
verifier reads final cash, survival days, and bankruptcy from that run's
`world.nmdb`, never from the agent's own accounting.
