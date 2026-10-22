"""Inference engines on another host, controlled and served over HTTP.

``python -m reef.inference.remote`` runs on the GPU host. It starts the engines
that the control plane asks for, then serves their control operations and their
inference requests over HTTP. On the control plane, ``inference.backend:
remote`` selects :class:`~reef.inference.remote.service.RemoteInferenceService`.
The trainer delivers weights as files (``reef-weight-files-v1``), and the
coordinator uploads them to the component. No tensor transport crosses the
network boundary.

This package imports no engine library at module scope; the component loads the
selected engine integration only when the control plane starts it.
"""
