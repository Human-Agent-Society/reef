"""The built-in SFT-then-RL recipe: SFT on held samples, then RL task jobs on the scenario's imported tasks.

- ``processor``: ``SFTThenRLJobData``, the sample-to-job stage that holds SFT samples and hands the imported train
  tasks to the training component as task jobs.
- ``recipe``: ``SFTThenRLRecipe``, the SFT stages with that sample-to-job stage, and the method of each job:
  SFT for trajectories, ``task_group`` for tasks.
"""

from reef.recipe.sft_then_rl.processor import SFTThenRLJobData, SFTThenRLJobSettings
from reef.recipe.sft_then_rl.recipe import SFTThenRLRecipe

__all__ = ["SFTThenRLJobData", "SFTThenRLJobSettings", "SFTThenRLRecipe"]
