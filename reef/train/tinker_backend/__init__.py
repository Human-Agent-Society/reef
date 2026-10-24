"""Optional Tinker integration: remote LoRA training, with sampling on Tinker or on Reef's serving engines.

Deployment/config discovery does not import the SDK. ``client`` owns the
training SDK boundary, ``losses`` names the Tinker loss and shapes its
inputs, and ``component`` is the training component that runs each job in
process behind ``reef.train.job_runtime.JobTrainingRuntime``. Sampling on
Tinker itself is ``reef.inference.tinker``; serving on engines is
``reef.runtime.serving``. Each side reads the checkpoint an artifact carries.
"""
