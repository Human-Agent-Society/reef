# Tutorials

One directory per tutorial. Each opens with a README that says what you will see, then a notebook that runs the whole pass cell by cell, then the script form. Configs and helper code sit in subdirectories. The recipe examples under [recipes/](../recipes/README.md) are the canonical run configs for each method; the tutorials are the walkthroughs.

| Tutorial | What you see | Needs | Open |
|----------|--------------|-------|------|
| [Evolve your harness](evolve-your-harness/README.md) | Three coding tasks go through Reef, one fails, the served model proposes a change to its own harness, the gate runs both versions, and the winner publishes as a new harness version. | An OpenAI compatible model endpoint (a local ollama is enough), no GPU; the pi coding agent for the pi variant, nothing extra for the native variant. | [notebook](evolve-your-harness/evolve-your-harness.ipynb), or `./run.sh` in the directory |
| [Reefine](reefine/README.md) | A person asks reef-pi for a capability in plain words; the service proposer writes the change, the gate scores it, an evolved extension waits for a promote, and the next session runs the new tree. Two demos (a bug fix flow, a research loop) and a measurement of which requests won the gate. | An OpenAI compatible model endpoint (a local ollama is enough), no GPU; the pi coding agent. | `./run.sh bugfix` in the directory |

Weight training walkthroughs are on the way; until then, [Evolve your model](../docs/user-guide/evolve-your-model.rst) and the recipe examples under [recipes/](../recipes/README.md) are the path.
