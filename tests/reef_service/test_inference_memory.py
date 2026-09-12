"""Backend-independent memory transitions reject uncertain partial operations."""

import pytest

from reef.runtime.inference_memory import InferenceMemory


class _Memory:
    def __init__(self):
        self.resident = {"weights", "kv"}
        self.failure = None

    def release(self, regions):
        for region in regions:
            self.resident.remove(region)
            if self.failure == "release":
                raise TimeoutError("partial release")

    def resume(self, regions):
        for region in regions:
            self.resident.add(region)
            if self.failure == "resume":
                raise TimeoutError("partial resume")


def test_memory_rejects_unknown_regions_before_mutating_the_engine():
    backend = _Memory()
    memory = InferenceMemory(backend, ("weights", "kv"))
    with pytest.raises(ValueError, match="unknown"):
        memory.release(["weights", "unknown"])
    assert backend.resident == {"weights", "kv"}
    memory.release(["kv", "kv"])
    memory.release(["kv"])
    assert backend.resident == {"weights"}
    memory.resume()
    memory.resume()
    assert backend.resident == {"weights", "kv"}


@pytest.mark.parametrize("failure", ["release", "resume"])
def test_partial_memory_failure_blocks_every_transition_until_replacement(failure):
    backend = _Memory()
    memory = InferenceMemory(backend, ("weights", "kv"))
    backend.failure = failure
    with pytest.raises(TimeoutError, match="partial"):
        memory.release()
        memory.resume()
    resident = set(backend.resident)
    for operation in (memory.release, memory.resume):
        with pytest.raises(RuntimeError, match="replace the engine"):
            operation()
    assert backend.resident == resident
    fresh = _Memory()
    replacement = InferenceMemory(fresh, ("weights", "kv"))
    replacement.release()
    replacement.resume()
    assert fresh.resident == {"weights", "kv"}
