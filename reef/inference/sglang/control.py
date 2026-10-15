"""SGLang inference control actor, independent of the training backend.

Engine handles and locks travel through control RPCs; weight tensors
continue to travel directly between training workers and inference engines.
"""

from __future__ import annotations

from reef.inference.engine_control import EngineControl
from reef.inference.sglang.config import SGLangConfig


class SGLangControl(EngineControl[SGLangConfig]):
    """Own the SGLang serving executor inside Reef's inference control actor."""
