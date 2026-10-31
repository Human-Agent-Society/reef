"""One Slime job in a process of its own: ``python -m reef.train.slime_backend.driver JOB_DIRECTORY``.

``SlimeTrainingComponent`` starts the driver for each job, with the job's
directory: ``job.json`` (the submitted job) and ``driver.json`` (the
deployment's Slime settings). A task job's driver, and a dataset job's, creates or attaches to the
component's Slime serving session, which holds the rollout engines and their
GPUs across jobs; a driver of another job does so while a session runs, and
otherwise places the job with Slime's own placement and starts no engines
(``--rollout-num-gpus=0``). The driver trains the job with ``SlimeJobRunner``,
which records its checkpoints in the directory, and writes the job's metrics to
``driver-result.json``. When it exits, also on SIGTERM, it releases its workers
and detaches from the session, or releases the GPUs it placed. A failed job
exits with a non-zero status and the error on its output.

``--dispose SAVE_ROOT RAY_ADDRESS NAMESPACE`` disposes the session that the
save root records: the component runs it when it closes.
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
    dispose_component_serving,
    settings_from_json,
    start_slime_runner,
)
from reef.train.slime_backend.rollout_group import SERVING_FILE


#: How long the disposal of the serving session may take, in seconds.
DISPOSE_TIMEOUT_S = 120.0


def main(argv: Sequence[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments[:1] == ["--dispose"] and len(arguments) == 4:
        root, ray_address, namespace = arguments[1:]
        dispose_component_serving(
            Path(root), ray_address=ray_address, namespace=namespace, timeout_s=DISPOSE_TIMEOUT_S
        )
        return
    if len(arguments) != 1:
        raise SystemExit(
            "usage: python -m reef.train.slime_backend.driver JOB_DIRECTORY\n"
            "       python -m reef.train.slime_backend.driver --dispose SAVE_ROOT RAY_ADDRESS NAMESPACE"
        )
    directory = Path(arguments[0])
    submitted = read_json(directory / JOB_FILE)
    settings_value = read_json(directory / DRIVER_FILE)
    if submitted is None or settings_value is None:
        raise SystemExit(f"{directory} holds no {JOB_FILE} and {DRIVER_FILE}")
    job = TrainingJob(
        directory.name,
        "slime",
        submitted["start"],
        JobData.from_dict(submitted["data"]),
        submitted["parameters"],
        scenario=submitted["scenario"],
    )
    settings = settings_from_json(settings_value)
    # A task job and a dataset job roll out on the serving session's engines; another job trains on the
    # session's GPUs while a session holds them, else on GPUs of its own, without engines.
    serving = (
        any(isinstance(item, TaskSample) for item in job.data.samples)
        or "dataset" in job.parameters
        or (directory.parent.parent / SERVING_FILE).exists()
    )
    if not serving:
        settings = settings_from_json({**settings_value, "arguments": [*settings.arguments, NO_ROLLOUT_ENGINES]})

    def stop(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(f"the Slime driver of job {job.job_id} stopped on signal {signum}")

    # SIGTERM unwinds through ``close``, which releases the workers, the engines and the GPUs.
    signal.signal(signal.SIGTERM, stop)
    runner = start_slime_runner(settings, serving=serving)
    try:
        metrics = runner.run(job, directory)
        write_json(directory / DRIVER_RESULT_FILE, {"metrics": dict(metrics)})
    finally:
        runner.close()


if __name__ == "__main__":
    main()
