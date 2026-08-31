from __future__ import annotations

import hashlib


_BLOCK_BYTES = 64


class ShakeStream:
    def __init__(self, *, domain: bytes, seed: bytes) -> None:
        if not isinstance(domain, bytes):
            raise TypeError("domain must be bytes")
        if not domain:
            raise ValueError("domain must be nonempty")
        if not isinstance(seed, bytes):
            raise TypeError("seed must be bytes")
        if len(seed) != 32:
            raise ValueError("seed must contain exactly 32 bytes")
        self._domain = domain
        self._seed = seed
        self._block_index = 0
        self._buffer = bytearray()

    def read(self, count: int) -> bytes:
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be nonnegative")
        while len(self._buffer) < count:
            block = hashlib.shake_256(
                self._domain
                + b"\0"
                + self._seed
                + self._block_index.to_bytes(8, "little")
            ).digest(_BLOCK_BYTES)
            self._buffer.extend(block)
            self._block_index += 1

        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result

    def randbelow(self, upper: int) -> int:
        if not isinstance(upper, int) or isinstance(upper, bool):
            raise TypeError("upper must be an integer")
        if upper <= 0:
            raise ValueError("upper must be positive")
        width = max(1, ((upper - 1).bit_length() + 7) // 8)
        sample_space = 1 << (8 * width)
        limit = (sample_space // upper) * upper
        while True:
            candidate = int.from_bytes(self.read(width), "little")
            if candidate < limit:
                return candidate % upper
