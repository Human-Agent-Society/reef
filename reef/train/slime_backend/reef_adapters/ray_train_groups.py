"""Compatibility imports for the former Ray-only training group module."""

from reef.train.slime_backend.reef_adapters.train_groups import (
    SlimeTrainGroup,
    create_train_groups,
    prepare_critic_args,
)

ReefRayTrainGroup = SlimeTrainGroup

__all__ = ["ReefRayTrainGroup", "SlimeTrainGroup", "create_train_groups", "prepare_critic_args"]
