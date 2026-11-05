"""Mechanics that training components share to cut a job's data into optimizer steps.

``schedule`` is the step schedule, ``signals`` what an objective computes from a
job's data, and ``helpers`` the conventional step counter. Each training
component owns its objectives and its job parameters
(``reef.train.slime_backend.objectives``, ``reef.train.tinker_backend.objectives``).
"""
