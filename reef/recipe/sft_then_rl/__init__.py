"""The built-in SFT-then-RL recipe: SFT on held samples, then RL task jobs on the scenario's imported tasks.

- ``processor``: ``SFTThenRLJobData``, the sample-to-job stage that holds SFT samples and hands the imported train
  tasks to the training component as task jobs, and ``SFTThenRLJobPlan``, which gives each job its method.
- ``recipe``: ``SFTThenRLRecipe``, the SFT stages with that sample-to-job stage and job plan.
"""

from reef.recipe.sft_then_rl.processor import SFTThenRLJobData, SFTThenRLJobPlan, SFTThenRLJobSettings
from reef.recipe.sft_then_rl.recipe import SFTThenRLRecipe

__all__ = ["SFTThenRLJobData", "SFTThenRLJobPlan", "SFTThenRLJobSettings", "SFTThenRLRecipe"]
