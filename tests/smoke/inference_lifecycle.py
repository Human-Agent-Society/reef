"""Fixtures and the control client of the lifecycle profile of ``tests.smoke.native_inference``.

The lifecycle profile creates a scenario and changes what it serves through Reef's public API. ``SeededWeights``
and ``NoTraining`` let its recipes start a scenario without a training run. ``ControlClient`` sends the control
requests and logs each one with its answer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from aiohttp import ClientSession
from tests.support.runtime_stubs import StubTrainingComponent

from reef.artifact import BASE_FILE, BaseRoot, DefaultStart
from reef.core.values import JsonValue
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


class ControlClient:
    """Control requests to one Reef app with its service token, each appended to a JSON-lines log.

    Values: ``session`` is an open aiohttp session without default headers; ``base_url`` is the app's
    ``http://host:port``; ``token`` is the app's service token, which goes only into the ``Authorization`` header,
    never into the log; ``log`` is a file in an existing directory. It receives one JSON object per request: the
    label, method, path and body of the request, the status and text of the answer, and ``sent`` and ``ended``,
    ``time.monotonic`` readings that order the request among the exchanges of the wire log.
    Purpose: the lifecycle profile reads and changes a scenario through the public API with ``request``.
    """

    def __init__(self, session: ClientSession, base_url: str, token: str, log: Path) -> None:
        self.session = session
        self.base_url = base_url
        self.token = token
        self.log = log

    async def request(
        self,
        label: str,
        method: str,
        path: str,
        body: dict[str, JsonValue] | None = None,
        *,
        status: int = 200,
    ) -> dict[str, JsonValue]:
        """Send one request, log it with its answer, and return the answer, which must have the HTTP ``status``.

        An answer with a ``status`` below 400 must be a JSON object. A refusal that the caller expects is plain
        text: the result is then ``{"error": <the answer's text>}``.
        """
        sent = time.monotonic()
        async with self.session.request(
            method, self.base_url + path, json=body, headers={"Authorization": f"Bearer {self.token}"}
        ) as response:
            answered, text = response.status, await response.text()
        line: dict[str, JsonValue] = {
            "label": label,
            "method": method,
            "path": path,
            "request_body": body,
            "status": answered,
            "response_text": text,
            "sent": sent,
            "ended": time.monotonic(),
        }
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, sort_keys=True) + "\n")
        assert answered == status, f"{label}: HTTP {answered}, expected {status}: {text[:400]!r}"
        if status >= 400:
            return {"error": text}
        try:
            answer = json.loads(text)
        except ValueError:
            answer = None
        assert isinstance(answer, dict), f"{label}: the answer is not a JSON object: {text[:400]!r}"
        return answer
