"""The bundled methods, one package each, laid out like stable-baselines3.

``recipes/<method>/`` holds a method's recipe, processor, preparer and (for
weight methods) its ``slime/`` backend half, plus ``examples/`` — the
runnable Harbor task + harness stacks that drive it. ``recipes/basic`` is the
example for the record-only base ``recipe`` kind, which has no method
package. ``reef/__init__`` imports the method packages, which registers
their kinds; the examples never ship in the wheel.
"""
