"""Process execution and isolation for harnesses and Harbor trials.

The ``bwrap`` and ``seatbelt`` packages hold the local sandboxes of Linux and macOS. The E2B and
Modal packages hold their executors and trial environments, and the boxd package holds its
executor. AgentENV provides image builds and Compose support for E2B-compatible trials. Shared
session, file-transfer and tunnel support lives in ``remote``.
"""
