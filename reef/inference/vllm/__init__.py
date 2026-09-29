"""vLLM inference capture behind Reef's runtime contracts.

This package holds the vLLM implementations of the contracts SGLang already
implements under :mod:`reef.inference.sglang`. It imports vLLM only inside
the adapters that need it, never a training backend. Serving and recording
are buffered in this version: a request that asks for a stream receives the
completed turn as one burst of protocol frames.
"""
