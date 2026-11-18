"""One retry of Harbor verification without running the agent again."""

import asyncio

from harbor.models.task.config import StepConfig
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from harbor.verifier.verifier import AddTestsDirError, RewardFileNotFoundError, Verifier


class RetryingVerifier(BaseVerifier):
    """A Harbor verifier that retries once when verification produced no reward file.

    Values: the inherited ``task`` is a Harbor ``Task``, ``trial_paths`` a
    ``TrialPaths``, and ``environment`` a ``BaseEnvironment``. ``override_env``
    maps variable names to strings; ``verifier_env`` is such a mapping or None.
    ``logger`` is a Python logger. ``step_name`` is a string or None;
    ``include_logs`` and ``exclude_logs`` are lists of path patterns or None.
    Harbor's base constructor initializes these fields. This subclass adds
    no state or construction invariants. Harbor's factory makes it for Reef's
    task trials, whose verification phase calls ``verify``.

    ``verify`` returns the first successful ``VerifierResult`` unchanged,
    including a zero reward. A failed test upload or missing reward file
    allows one further call after one second, on the same environment.
    Other failures and a failed second call propagate. Harbor's enclosing
    timeout bounds both calls and the delay. No agent or environment starts
    here. Empty or invalid reward files are not retried, so their contents
    cannot mask the next attempt's reward. A step name absent from the task
    raises ``ValueError`` before verification starts.
    """

    async def verify(self) -> VerifierResult:
        step: StepConfig | None = None
        if self.step_name is not None:
            step = next((step for step in self.task.config.steps or () if step.name == self.step_name), None)
            if step is None:
                raise ValueError(f"task has no verifier step {self.step_name!r}")
        definition = resolve_verifier_environment_definition(self.task.config, self.task.paths, step)
        # Harbor's custom-verifier factory does not pass its skip_tests_upload argument.
        # Only a verifier's own image bundles its tests; reusing the task image needs an upload.
        verifier = Verifier(
            task=self.task,
            trial_paths=self.trial_paths,
            environment=self.environment,
            override_env=self.override_env,
            logger=self.logger,
            skip_tests_upload=definition is not None and definition.bundled_tests,
            verifier_env=self.verifier_env,
            step_name=self.step_name,
            include_logs=self.include_logs,
            exclude_logs=self.exclude_logs,
        )
        try:
            return await verifier.verify()
        except (AddTestsDirError, RewardFileNotFoundError) as error:
            self.logger.warning("Verifier produced no reward; retrying once in the same environment: %s", error)
        await asyncio.sleep(1.0)
        return await verifier.verify()
