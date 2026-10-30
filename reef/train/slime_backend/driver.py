"""One Slime job in a process of its own: ``python -m reef.train.slime_backend.driver JOB_DIRECTORY``.

``SlimeTrainingComponent`` starts the driver for each job, with the job's
directory: ``job.json`` (the submitted job) and ``driver.json`` (the
deployment's Slime settings). The driver places the job with Slime's own
placement, starts Slime's rollout engines when the job is a task job (a job of
any other data starts none, ``--rollout-num-gpus=0``), trains the job with
``SlimeJobRunner``, which records its checkpoints in the directory, and writes
the job's metrics to ``driver-result.json``. It releases its workers, engines
and GPUs when it exits, also on SIGTERM. A failed job exits with a non-zero
status and the error on its output.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from reef.core.job_data import JobData, TaskSample
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.jobs import TrainingJob
from reef.train.local_jobs import JOB_FILE
from reef.train.slime_backend.component import (
    DRIVER_FILE,
    DRIVER_RESULT_FILE,
    NO_ROLLOUT_ENGINES,
    settings_from_json,
    start_slime_runner,
)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) != 1:
        raise SystemExit("usage: python -m reef.train.slime_backend.driver JOB_DIRECTORY")
    directory = Path(arguments[0])
    submitted = read_json(directory / JOB_FILE)
    settings_value = read_json(directory / DRIVER_FILE)
    if submitted is None or settings_value is None:
        raise SystemExit(f"{directory} holds no {JOB_FILE} and {DRIVER_FILE}")
    job = TrainingJob(
        directory.name, "slime", submitted["start"], JobData.from_dict(submitted["data"]), submitted["parameters"]
    )
    settings = settings_from_json(settings_value)
    if not any(isinstance(item, TaskSample) for item in job.data.samples):
        settings = settings_from_json({**settings_value, "arguments": [*settings.arguments, NO_ROLLOUT_ENGINES]})

    def stop(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(f"the Slime driver of job {job.job_id} stopped on signal {signum}")

    # SIGTERM unwinds through ``close``, which releases the workers, the engines and the GPUs.
    signal.signal(signal.SIGTERM, stop)
    runner = start_slime_runner(settings)
    try:
        metrics = runner.run(job, directory)
        write_json(directory / DRIVER_RESULT_FILE, {"metrics": dict(metrics)})
    finally:
        runner.close()


if __name__ == "__main__":
    main()
