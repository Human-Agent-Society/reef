# Run your own LangGraph agent

This example connects a LangGraph calculator through Reef's existing `python`
adapter. The graph calls a model, executes `multiply`, calls the model again,
and writes `answer.txt`. Harbor checks the answer and the tool call in ATIF.

The agent owns its graph and tools. Reef supplies its model connection and
records the calls. The deployment is record-only: this example does not send
training reports, evolve the program, or publish a harness version.

## Requirements

- Python 3.12 or newer, uv, and a current Reef checkout with the Python adapter
  and Python-to-ATIF conversion.
- Docker running, with at least 1 GiB available for the task.
- Network access for the container image, uv, Python, and the pinned packages.
- For the real-model run: an OpenAI Chat Completions endpoint with tool calling.

From the repository root, reuse your Reef environment. For a new checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

LangGraph and LangChain are installed inside the task container from the
inline dependencies in `tree/program/main.py`. They are not Reef dependencies.

## Run without an API key

From the repository root:

```bash
python tutorials/custom-agent/run.py --smoke
```

`--smoke` starts a local scripted provider. It requests `multiply(17, 23)`,
then returns the tool's result. Reef, LangGraph, the tool, Docker, ATIF
conversion, and the verifier all run. This checks the integration, not model
quality. The driver starts and stops its own Reef service on a temporary port.

Expected final line:

```text
Passed: model -> multiply tool -> model -> answer.txt -> verifier reward 1.0
```

The printed JSON has reward `1.0`, at least two receipts, and an empty `reports`
list. A failed command, missing model calls, or a failed task makes the driver
exit nonzero. A first run takes longer while Docker and uv download dependencies.

Docker Desktop uses `host.docker.internal` by default. On Linux, pass the
Docker bridge address or another host address reachable from the container:

```bash
python tutorials/custom-agent/run.py --smoke --agent-host 172.17.0.1
```

The bridge address can differ; inspect your Docker network before using it.

## Run with a model

Set your provider credential in `REEF_UPSTREAM_API_KEY`, then run:

```bash
python tutorials/custom-agent/run.py \
  --upstream-url https://your-provider.example \
  --model your-tool-capable-model
```

Replace both values. The upstream URL excludes `/v1`; Reef supplies the
agent's `/v1` URL. For a local endpoint without authentication, leave
`REEF_UPSTREAM_API_KEY` unset. The model must support Chat Completions, tool
calling, and `max_tokens`. No streaming or Responses API is used here.

The real model chooses its tool calls and answer. A reward of zero is a failed
task, not necessarily a connection error. Inspect the transcript to distinguish
the two. Do not use `--smoke` with the real-provider options.

## Inspect a run

Each invocation prints a fresh directory under `tutorials/custom-agent/work/`:

| File | What it contains |
| --- | --- |
| `result.json` | Reward, receipt IDs, trial URI, and whether replies were scripted |
| `reef.log` | Service startup and runtime logs |
| `records/` | Reef's persisted calls |
| `tree/program/main.py` | The source program used for this run |
| Trial `agent/command.json` | Agent exit code, stdout, and stderr |
| Trial `agent/sessions/calls.jsonl` | Request/response bodies, without HTTP headers |
| Trial `agent/trajectory.json` | ATIF supplied to the verifier |
| Trial `verifier/reward.txt` | Task score |

The trial location is `trial_uri` in `result.json`. Keep run directories private:
they contain prompts and model outputs. The program's rendered model binding
is runtime configuration; do not copy it into the source tree.

If Reef cannot start, read `reef.log`. If the container cannot reach the model,
check `--agent-host`. Package installation errors appear in the trial's setup
logs. Model errors and the graph's step limit appear in `agent/command.json`.

## Adapt your own agent

1. Replace the graph and tools in `tree/program/main.py`. Keep its task argument.
2. Keep reading `model.json` beside the program, and bind all model nodes to it.
3. Pin dependencies in the script metadata. Keep source modules flat under `program/`.
4. Keep JSONL request/response logging if a verifier needs ATIF. The synchronous
   `SessionTransport` is specific to this example's non-streaming HTTP client.
5. Replace `harbor/` with your task and verifier. Adjust the success checks in
   `run.py` to your task's score and expected calls.

The program runs in the task's working directory. It imports no Reef service
or training code. `run.py` is the host-side launcher; it uses `tree_agent` and
`TaskRolloutRunner` to reuse Reef's rendering, capture, and Harbor integration.

See [Connect your own agent](../../docs/user-guide/custom-agents.rst) for the
other entry points and the boundary between task scores, feedback, and evolution.

## Test the agent without Docker

These tests run the real graph against the scripted HTTP provider, convert its
logs to ATIF, and check a model error. Run them from the repository root:

```bash
uv run --no-project --with-editable '.[dev]' \
  --with 'langgraph==1.2.19' --with 'langchain-openai==1.6.0' \
  python -m pytest tests/harness/test_custom_agent_example.py -q -o addopts=''
```

The graph tests skip when their optional packages are absent. The Docker smoke
above additionally checks the actual Reef service, adapter installation, and
Harbor verifier lifecycle.
