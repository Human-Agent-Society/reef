# Tinker training smoke

This runs four text completions (at most 32 new tokens each), assigns **synthetic** rewards 0, 1, 2, and 3, and waits for one TTTD training commit. It checks the inference → feedback → training → publication mechanism, not model quality. Both serving startup and the smoke use the paid Tinker API.

Four rollouts keep TTTD's adaptive-entropic advantages moderate for these rewards. With only two distinct rewards, its fixed `log(2)` KL target reaches the maximum possible concentration, and the leave-one-out normalization can produce advantages near `1e12`.

Use Python 3.11+ on Linux or macOS, Git LFS, and a Tinker account with access to the configured model. From the Reef checkout:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[tinker]' -e ./third_party/reef-client
# Supply TINKER_API_KEY through your shell/secret manager, never in serve.yaml.
export REEF_TOKEN=reef-local
export REEF_TINKER_STATE_DIR="$PWD/work/tinker-smoke"
python -m reef serve -c tutorials/tinker/serve.yaml
```

After `/healthz` is ready, run in another terminal with the same virtual environment and `REEF_TOKEN`:

```bash
python tutorials/tinker/smoke.py
```

The service owns one training scenario. Run the smoke once per fresh service/state directory; keep that state directory to inspect or resume the resulting release. The randomly named scenario prevents old reports from being reused accidentally, but does not enable concurrent scenarios.

The model name is sent to Tinker; Reef downloads only the tokenizer resources needed by the SDK, not the full model. Select a model available to your account. In a wheel-only installation, supply your own recipe package; this checkout's `recipes.tttd` is not bundled in `reef-infra`.

Each version stores a `tinker-checkpoint.json` manifest in Reef's artifact repository. It points to durable Tinker training (including Adam state) and sampler checkpoints. Keep the artifact/record directories **and** the Tinker checkpoints; local artifact retention does not delete remote checkpoints. Access to the originating Tinker project remains necessary after restart or rollback.

See [Tinker backend](../../docs/user-guide/tinker.rst) for supported requests, loss adapters, recovery semantics, and current limits. The existing TTTD benchmark `run.sh` configures the Slime GPU stack; this smaller smoke has its own configuration and runner.
