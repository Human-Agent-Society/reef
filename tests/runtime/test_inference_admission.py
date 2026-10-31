"""Inference admission: a closed runtime admits no new request and lets admitted ones finish."""

from __future__ import annotations

import asyncio

import pytest

from reef.core.errors import InferenceUnavailable
from reef.runtime.interfaces import InferenceAdmissionController


def test_closed_admission_refuses_new_requests_and_lets_admitted_ones_finish() -> None:
    async def run() -> None:
        controller = InferenceAdmissionController()
        admitted = await controller.acquire()
        controller.close()
        with pytest.raises(InferenceUnavailable, match="shutting down"):
            await controller.acquire()
        assert controller.status == {"open": False, "active": 1}
        admitted.release()
        assert controller.status == {"open": False, "active": 0}

    asyncio.run(run())
