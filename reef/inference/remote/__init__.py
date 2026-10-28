"""Inference engines on another host, controlled and served over HTTP.

``python -m reef.inference.remote`` runs on the GPU host. It starts the engines
that the control plane asks for, then serves their control operations and their
inference requests over HTTP. On the control plane, an ``inference.remote``
section runs the engines that ``inference.backend`` names on the component,
through :class:`~reef.inference.remote.service.RemoteInferenceService`.
The control plane uploads each published release's weight files to the
component. No tensor transport crosses the network boundary.

This package imports no engine library at module scope; the component loads the
selected engine integration only when the control plane starts it.
"""
