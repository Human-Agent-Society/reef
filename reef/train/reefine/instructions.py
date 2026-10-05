"""Instructions supplied by the Reefine recipe to its proposal machinery."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProposalInstructions:
    templates: Mapping[str, str]
