"""The built-in supervised fine-tuning (SFT) recipe: train the served model on target responses.

A deployment selects the recipe as ``reef.recipe.sft:SFTRecipe``. Its
``pipeline`` section configures the two stages.

- ``processor``: the stages. ``SFTSamples`` makes each report its own
  record set, never drops a sample as stale, and renders the recorded request
  and the target response with the served model's chat template. ``SFTJobData`` puts the tasks into job data, together with
  uploaded ATIF trajectories, which ``SFTSamples`` renders with every agent
  turn trained.
- ``recipe``: the default objectives and the report contract
  (``reef.core.reports.TeacherContextReport``).

The ``sft`` objective belongs to each training backend that supports it
(``reef.train.slime_backend.objectives``, ``reef.train.tinker_backend.objectives``).
Another recipe can select it for a job with the job parameter ``objective``.
"""

from reef.recipe.sft.processor import SFTJobData, SFTSamples, SFTSettings
from reef.recipe.sft.recipe import SFTRecipe

__all__ = ["SFTJobData", "SFTRecipe", "SFTSamples", "SFTSettings"]
