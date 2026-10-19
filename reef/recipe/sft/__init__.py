"""The built-in supervised fine-tuning (SFT) recipe: train the served model on target responses.

A deployment selects the recipe as ``reef.recipe.sft:SFTRecipe``. Its
``pipeline`` section configures the three stages.

- ``processor``: the stages. ``SFTRecord2Record`` makes each report its own
  record set and never drops a set as stale. ``SFTTasks`` renders the
  recorded request and the target response with the served model's chat
  template. The shared ``DistillBatches`` stage puts the tasks into batches.
- ``objective``: ``SftObjective`` trains all samples with the same weight.
  Another recipe can select it for a job with
  ``TrainingMethod("reef.recipe.sft:SftObjective")``.
- ``recipe``: the objective, the step schedule, and the report contract
  (``reef.core.reports.TeacherContextReport``).

The ``sft`` loss family is in ``reef.train.slime_backend.sft`` and uses
Slime's ``sft_loss``. This package registers the family by reference. Thus
only the training driver and the workers import the Slime code.
"""

from reef.recipe.sft.objective import SftObjective
from reef.recipe.sft.processor import SFTRecord2Record, SFTSettings, SFTTasks
from reef.recipe.sft.recipe import SFTRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sft", "reef.train.slime_backend.sft:SftAlgorithm")

__all__ = ["SFTRecipe", "SFTRecord2Record", "SFTSettings", "SFTTasks", "SftObjective"]
