"""Durable training-step coordination and commit-gated weight publication.

Backends retain admission, data preparation, optimizer execution, checkpoint I/O
and tensor transport. Reef owns job identity/replay, RUNNING/CHECKPOINT ordering,
publication ordering, unchanged-weight republication and fenced startup recovery without
importing a concrete model framework.
"""
