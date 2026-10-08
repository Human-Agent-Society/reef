"""vLLM inference capture behind Reef's runtime contracts.

This package holds the vLLM implementations of the contracts SGLang already
implements under :mod:`reef.inference.sglang`: request capture, the KV
connector that stamps tokens with their weight version, and engine launch and
control over vLLM's own HTTP routes. It imports vLLM only inside the connector,
which runs in the engine process, and never a training backend. Serving and
recording are buffered in this version: a request that asks for a stream
receives the completed turn as one burst of protocol frames.
"""
