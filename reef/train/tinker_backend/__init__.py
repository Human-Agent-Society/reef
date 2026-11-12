"""Optional Tinker integration: remote LoRA training, with sampling on Tinker or on Reef's serving engines.

Deployment/config discovery does not import the SDK. ``component`` is the training component that runs each job
in process (``reef.train.local_jobs.LocalJobComponent``): ``parameters`` parses the job's parameters, ``objectives`` names its method, ``signals`` is what the method computes, ``schedule`` is the step schedule that
``preparation`` cuts the job's data into optimizer steps with, ``rollout`` runs a task job's rollouts, ``losses`` names each step's Tinker loss and shapes its inputs, ``loss_families`` holds the losses by name, and
``client`` is the training SDK boundary. ``facts`` is what each backend states from its settings without a client
(``reef.train.component.TrainingBackendFacts``). The SDK pieces that training and serving share, the checkpoint manifest,
the SDK session and sampling, are ``reef.tinker``. Sampling on Tinker itself is ``reef.inference.tinker``; serving
on engines is ``reef.inference.serving``. Each side reads the checkpoint a snapshot carries.
"""
