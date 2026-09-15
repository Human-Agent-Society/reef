"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

- ``environment_loader``: load and play a generated environment as the reference code does; ships into every task.
- ``tasks``: a generated environment as a Harbor task directory, with a replay verifier and a split per generation.
- ``designer``: the Environment Designer's adversarial prompt, its reply parsed, and the smoke test of what it wrote.
- ``play``: the Reasoning Agent plays one environment turn by turn, the game in a child process.

The regeneration step that ties them together and the training side follow.
"""

from recipes.beta.spade.designer import (
    DesignerReply,
    DesignerReplyError,
    DesignerRequest,
    PlayRecord,
    SmokeResult,
    designer_messages,
    designer_prompt,
    parse_designer_reply,
    smoke_test,
)
from recipes.beta.spade.environment_loader import (
    environment_class_name,
    episode_return,
    extract_boxed_answer,
    load_environment_class,
    make_environment,
    normalized_action,
    replay,
)
from recipes.beta.spade.play import Episode, GameProcess, GameProcessError, Turn, gameplay_messages, play_episode
from recipes.beta.spade.tasks import GeneratedEnvironment, environment_task, split_generation

__all__ = [
    "DesignerReply",
    "DesignerReplyError",
    "DesignerRequest",
    "Episode",
    "GameProcess",
    "GameProcessError",
    "GeneratedEnvironment",
    "PlayRecord",
    "SmokeResult",
    "Turn",
    "designer_messages",
    "designer_prompt",
    "environment_class_name",
    "environment_task",
    "episode_return",
    "extract_boxed_answer",
    "gameplay_messages",
    "load_environment_class",
    "make_environment",
    "normalized_action",
    "parse_designer_reply",
    "play_episode",
    "replay",
    "smoke_test",
    "split_generation",
]
