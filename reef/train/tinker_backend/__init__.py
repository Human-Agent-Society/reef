"""Optional Tinker integration: remote LoRA training, with sampling on Tinker or on Reef's serving engines.

Deployment/config discovery does not import the SDK. ``component`` is the training component that runs each job
in process (``reef.train.local_jobs.LocalJobComponent``): ``preparation`` cuts the job's data into optimizer
steps, ``rollout`` runs a task job's episodes, ``losses`` names each step's Tinker loss and shapes its inputs, and
``client`` is the training SDK boundary. The SDK pieces that training and serving share, the checkpoint manifest,
the SDK session and sampling, are ``reef.tinker``. Sampling on Tinker itself is ``reef.inference.tinker``; serving
on engines is ``reef.runtime.serving``. Each side reads the checkpoint a release carries.
"""
