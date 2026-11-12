"""Fixtures of the lifecycle profile of ``tests.smoke.native_inference``.

The lifecycle profile creates a scenario and changes what it serves through Reef's public API. ``SeededWeights``
and ``NoTraining`` let its recipes start a scenario without a training run.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.runtime_stubs import StubTrainingComponent

from reef.artifact import BASE_FILE, BaseRoot, DefaultStart
from reef.train.harness_backend.contract import Method, MethodJob


class SeededWeights(StubTrainingComponent):
    """A training component whose untrained checkpoint is an existing model directory.

    Values: ``directory`` is an existing directory that holds the files of one full-weight checkpoint; the
    constructor raises ``ValueError`` for another path.
    Purpose: the facts of the component (``StubFacts``) write the root snapshot of a new scenario of the lifecycle
    profile's weight recipe: a base root that names ``directory`` as the base model. The serving runtime has
    ``directory`` as its model path, so it loads that directory for the root. The component trains nothing: a stub
    without a checkpoint root fails each submitted job.
    """

    def __init__(self, directory: Path) -> None:
        super().__init__()
        if not directory.is_dir():
            raise ValueError(f"the checkpoint directory {directory} does not exist")
        self.directory = directory

    def stub_default_start(self) -> DefaultStart:
        model = str(self.directory)
        return DefaultStart(files={BASE_FILE: BaseRoot({"model": model}).to_text()}, inputs={"model": model})


class NoTraining(Method):
    """A harness evolution method that must never run a job.

    Values: it holds no state and takes no configuration.
    Purpose: the lifecycle profile's harness training component needs a method. The profile imports and serves harness
    artifacts without an evolution job, so a job fails the run.
    """

    def run(self, job: MethodJob) -> None:
        raise AssertionError("The lifecycle qualification must not train")
