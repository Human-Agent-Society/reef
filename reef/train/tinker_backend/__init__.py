"""Optional Tinker integration: remote LoRA training, with sampling on Tinker or a local engine.

Deployment/config discovery does not import the SDK. ``client`` owns the
training SDK boundary, ``losses`` names the Tinker loss and shapes its
inputs, ``runtime`` is the in-process training runtime, and ``backend`` and
``training`` are the same trainer behind Reef's coordinator for a local
SGLang engine. Sampling on Tinker itself is ``reef.inference.tinker``; the
two sides share only the checkpoint manifest an artifact carries.
"""
