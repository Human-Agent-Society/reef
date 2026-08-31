"""The SkillClaw method package: what skillclaw.yaml's dotted references name.

``harness.skillclaw`` supplies the mechanism's three callables - ``propose``
(the sealed night flow mapped to one composite mutation sequence),
``evaluate`` (exact-answer probe grading), and the ``selection: always``
policy set in skillclaw.yaml. ``harness.skillclaw_recipe`` is the recipe
subclass the yaml boots (``harness.skillclaw_recipe:SkillClawRecipe``); its
surface (``harness.catalog``) injects the served pool's catalog into every
proxied request. The remaining modules are the sealed campaign's night
internals (``night``, ``evolver``, ``sessions``, ``prompts``), the
preregistered gain criterion (``stats``), and the shared constants
(``config``).

Unlike the sibling examples' harnesses this package imports ``reef`` itself,
not just ``reef_client``: the method runs inside the embedded Reef service
(run.py), not beside it.
"""
