"""SkillLibrary as a Reef harness evolution method: procedural skills induced from scored runs and kept after a replay.

A deployment names ``recipes.skill_library:SkillLibrary`` in ``evolution.method``. ``method`` holds the method,
``prompts/`` the extractor prompt, and ``examples/serve.yaml`` a deployment that replays evaluation tasks on pi.
"""

from recipes.skill_library.method import SkillLibrary

__all__ = ["SkillLibrary"]
