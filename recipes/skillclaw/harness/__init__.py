"""The SkillClaw harness package: what skillclaw.yaml's callable references name.

``harness.plugins`` holds the two classes skillclaw.yaml names:
``NightProposer`` (the sealed night flow mapped to one patch of the skill
pool) and ``ProbeScorer`` (exact-answer probe grading); both delegate to
``harness.skillclaw``. skillclaw.yaml sets ``selection: always``. The
cookbook recipe entry point lives at ``recipes.skillclaw.recipe``; its
surface uses this package's ``catalog`` module to inject the served pool's
catalog into every proxied request. The remaining modules are the sealed
campaign's night internals (``night``, ``evolver``, ``sessions``,
``prompts``), the preregistered gain criterion (``stats``), and the shared
constants (``config``).

Unlike the sibling examples' harnesses this package imports ``reef`` itself,
not just ``reef_client``: the method runs inside the embedded Reef service
(run.py), not beside it.
"""
