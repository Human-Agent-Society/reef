"""Compatibility import for the backend-independent runtime adapter."""

from reef.train.runtime_backend import RuntimeTrainingBackend

SlimeTrainingBackend = RuntimeTrainingBackend

__all__ = ["SlimeTrainingBackend"]
