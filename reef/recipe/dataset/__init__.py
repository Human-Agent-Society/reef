"""The built-in dataset recipe: train the served model with Slime on a prompt dataset, Slime-native.

A deployment selects the recipe as ``reef.recipe.dataset:DatasetRecipe`` on
the Slime backend, with ``data.training_mode: manual``. Each training
request (``POST /reef/train``) whose ``parameters`` hold ``dataset``
becomes one dataset job (``reef.train.slime_backend.dataset``): Slime rolls the
dataset's prompts out, rewards them and trains them with its own options in
``training.options``.

- ``processor``: ``DatasetJobData``, the job of one request, with no samples.
- ``recipe``: the default objective and the stage.

The ``slime_dataset`` objective belongs to Slime
(``reef.train.slime_backend.objectives``). Its ``slime`` loss family is in
``reef.train.slime_backend.native``; it leaves the training signal to Slime.
"""

from reef.recipe.dataset.processor import DatasetJobData
from reef.recipe.dataset.recipe import DatasetRecipe

__all__ = ["DatasetJobData", "DatasetRecipe"]
