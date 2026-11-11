# Tutorials

One directory per tutorial. Each opens with a README that says what you will see and how to run it. Some also provide a notebook. Configs and helper code sit in subdirectories. The recipe examples under [recipes/](../recipes/README.md) are the canonical run configs for each method; the tutorials are the walkthroughs.

| Tutorial | What you see | Needs | Open |
|----------|--------------|-------|------|
| [Connect your own agent](custom-agent/README.md) | A LangGraph agent runs a tool through Reef and completes a Harbor task with a recorded trajectory. | Docker and uv; scripted smoke replies or an OpenAI compatible model with tool calling. | `python tutorials/custom-agent/run.py --smoke` from the repository root |
| [Evolve your harness](evolve-your-harness/README.md) | Three coding tasks go through Reef, one fails, the served model proposes a change to its own harness, an evaluation runs both trees on the same tasks, and the winner publishes as a new harness version. | An OpenAI compatible model endpoint (a local ollama is enough), no GPU; the pi coding agent for the pi variant, nothing extra for the native variant. | [notebook](evolve-your-harness/evolve-your-harness.ipynb), or `./run.sh` in the directory |

The [Tinker smoke](tinker/README.md) covers remote weight training with four short rollouts and one update. For full training experiments, [Evolve your model](../docs/user-guide/evolve-your-model.rst) and the recipe examples under [recipes/](../recipes/README.md) are the path.
