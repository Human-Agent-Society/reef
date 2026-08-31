"""The skill-artifact surface: how the served text tree reaches its consumer.

A *skill artifact* is the versioned tree of skill text that reef serves to
its consumer — injected into requests or pulled from ``GET /reef/harness``.
This surface delivers the artifact; evolution runs in a method package, where
a skill may be one node kind of its composition tree.
"""

from reef.surface.skills.modules import RequestSkillLayer, SkillLayer
from reef.surface.skills.surface import SkillInferenceHooks, SkillValidator, create_skill_surface, validate_tree

__all__ = [
    "RequestSkillLayer",
    "SkillInferenceHooks",
    "SkillLayer",
    "SkillValidator",
    "create_skill_surface",
    "validate_tree",
]
