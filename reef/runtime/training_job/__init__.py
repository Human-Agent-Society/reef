"""Durable training-job markers and commit-gated weight publication.

Backends retain checkpoint production and tensor transport. This package owns
publication ordering and recovery without importing a concrete model framework.
"""
