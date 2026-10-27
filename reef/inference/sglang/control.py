"""SGLang inference control actor, independent of the training backend.

Engine handles travel through control RPCs; weights reach the engines as
files that the engines read themselves.
"""

from __future__ import annotations

from reef.inference.engine_control import EngineControl
from reef.inference.sglang.config import SGLangConfig


class SGLangControl(EngineControl[SGLangConfig]):
    """Own the SGLang serving executor inside Reef's inference control actor."""
