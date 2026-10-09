# AgentCL coding

One self-contained example supports `--method opd`, `sdft`, or `sdpo`.
Use Python 3.12 and install the example with `pip install -e .`.

```bash
./run.sh export --data-root /path/to/data --cache-dir /path/to/cache
AGENTCL_EXECUTION_APPROVED=1 ./run.sh verify-references --data-root /path/to/data
./run.sh train --method sdft --profile smoke --data-root /path/to/data --run-root /path/to/run --dry-run
```

The dry-run prints the native serving command. Start it under an approved
external supervisor, then run `baseline`, `train`, and `evaluate` with the
same method, scenario, data root and run root. `evaluate --phase independent`
uses the held-out split. OPD also requires `--teacher-checkpoint` with matching
model architecture and token IDs. Full runs use 96 ordered training tasks.

Training retains every assistant token and masks tool/context tokens. SDPO
samples all siblings before reporting; each task waits for its native commit.
Interrupted or repeated phases stop and are never automatically replayed.
Use a fresh scenario and run root after a failed phase.

W&B uploads require `AGENTCL_UPLOADS_APPROVED=1`, `--wandb-project` and
`--wandb-entity`. Apply the dry-run's observability override to the native
service for optimizer metrics; driver flags upload only evaluation summaries.
The example allocates no GPUs and creates no credentials.

Previous native runs proved all three methods can train. This consolidated
path has CPU tests; it has not been GPU-retested. Full benchmark gains and
numerical parity remain unverified.
