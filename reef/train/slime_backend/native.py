"""The ``slime`` loss family: Slime computes the training signal of a dataset job itself.

A dataset job (``reef.train.slime_backend.dataset``) carries no Reef training
signal. Slime rolls the dataset's prompts out, scores them with its reward
model (``--custom-rm-path`` or ``--rm-type``), computes the advantages with its
``--advantage-estimator`` and trains its ``policy_loss``. Slime's own hooks
customize the rest, for example ``--rollout-data-postprocess-path``, which sees
the rollout data after the advantages and before the loss. The built-in dataset
recipe (``reef.recipe.dataset``) registers this family by reference, so only
the training driver and the workers import this module.
"""

from __future__ import annotations

from argparse import Namespace

from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family

#: The data source that serves a dataset job's prompts to Slime's rollout manager.
DATASET_SOURCE_PATH = "reef.train.slime_backend.dataset_source.ReefDatasetSource"
#: The data source that Slime's parser selects when none is named.
SLIME_DEFAULT_SOURCE_PATH = "slime.data.data_source.RolloutDataSourceWithBuffer"


@register_loss_family
class SlimeNativeAlgorithm(SlimeAlgorithm):
    """Slime's ``policy_loss`` on the advantages that Slime computes from its own rollouts' rewards."""

    loss_family = "slime"
    loss_type = "policy_loss"
    advantages = "forbidden"
    forbidden_advantages_message = "the slime family trains dataset jobs, whose advantages Slime computes"
    allows_slime_advantage_computation = True
    # Slime's own engines roll a dataset job out and record the log-probabilities of each sampled token.
    accepts_rollout_logprobs = True

    def configure_backend_args(self, args: Namespace) -> None:
        # The serving session's rollout manager outlives the drivers, so the data source is the deployment's.
        if args.data_source_path not in (None, SLIME_DEFAULT_SOURCE_PATH, DATASET_SOURCE_PATH):
            raise ValueError("the slime family serves each dataset job's prompts itself; remove --data-source-path")
        args.data_source_path = DATASET_SOURCE_PATH

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not args.compute_advantages_and_returns:
            raise RuntimeError(f"{source} needs Slime's advantages: remove --disable-compute-advantages-and-returns")


__all__ = ["DATASET_SOURCE_PATH", "SlimeNativeAlgorithm"]
