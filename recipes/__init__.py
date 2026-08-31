"""Cookbook methods, one package each.

``recipes/<method>/`` holds a method's recipe, processor, preparer and (for
weight methods) its ``slime/`` backend half, plus ``examples/`` — the
runnable Harbor task + harness stacks that drive it. ``recipes/basic`` is the
example for the core record-only ``recipe``, which has no method package.
Reef does not import this tree, and none of it ships in the wheel. Example
deployments select recipe and loss classes through explicit dotted references.
"""
