"""vLLM inference capture behind Reef's inference contracts.

This package holds the vLLM implementations of the contracts SGLang already
implements under :mod:`reef.inference.sglang`: request capture, and engine
launch and control over vLLM's own HTTP routes. It imports vLLM only to build a
tool-call parser, and never a training backend. Serving and
recording are buffered in this version: a request that asks for a stream
receives the completed turn as one burst of protocol frames.
"""
