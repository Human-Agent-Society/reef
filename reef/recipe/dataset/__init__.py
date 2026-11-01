"""The built-in dataset recipe: train the served model with Slime on a prompt dataset, Slime-native.

A deployment selects the recipe as ``reef.recipe.dataset:DatasetRecipe`` on
the Slime backend, with ``data.training_mode: manual``. Each training
instruction (``POST /reef/train``) whose ``parameters`` hold ``dataset``
becomes one dataset job (``reef.train.slime_backend.dataset``): Slime rolls the
dataset's prompts out, rewards them and trains them with its own options in
``training.options``.

- ``objective``: ``SlimeDatasetObjective``, of the ``slime`` loss family,
  which leaves the training signal to Slime.
- ``processor``: ``DatasetJobData``, the job of one instruction, with no samples.
- ``recipe``: the objective and the stage.

The ``slime`` loss family is in ``reef.train.slime_backend.native``. This
package registers the family by reference, so only the training driver and
the workers import the Slime code.
"""

from reef.recipe.dataset.objective import SlimeDatasetObjective
from reef.recipe.dataset.processor import DatasetJobData
from reef.recipe.dataset.recipe import DatasetRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("slime", "reef.train.slime_backend.native:SlimeNativeAlgorithm")

__all__ = ["DatasetJobData", "DatasetRecipe", "SlimeDatasetObjective"]
