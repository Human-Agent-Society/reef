"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor; everything SPADE needs beyond it lives here. The Designer writes a
Harbor task, an instruction, a container, a verifier and a reference solution, and every task is one any
Harbor agent can play.

- ``designer``: the Environment Designer's adversarial prompt, its reply parsed, and the experience the prompt
  carries.
- ``harbor``: the written task under the team's structural gate, its hash, a split per generation, and Harbor's
  oracle check.
- ``generation``: one generation end to end: the Designer proposes through Reef, the oracle check refuses, the
  task is written, the Reasoning Agent plays both arms through the task player, regret splits, the manifest and the
  report are written, and each proposal is reported against the Designer's receipt.
- ``recipe``, ``processor``, ``preparer``: the Reasoning Agent's training: the task player's reports grouped by task,
  group relative advantages, Tinker's importance sampling loss.
- ``roles``: a role names its service, scenario, model, harness and reward; the deployment behind the
  scenario decides whether it evolves weights or a harness.
- ``designer_processor``, ``recipe``: the Designer's weight training: its proposals grouped by generation,
  regret as the reward, the same preparer and loss.
- ``harness``: the Designer's harness evolution: its prompt as a harness tree, rewritten from the regret
  reports once per generation, published without a gate.
- ``rounds``: Designer rounds then a Reasoning Agent round, the other side fixed, every report naming its
  opponent's version.
"""

from recipes.beta.spade.designer import (
    DesignerReplyError,
    DesignerRequest,
    HarborReply,
    PlayRecord,
    designer_messages,
    designer_prompt,
    parse_harbor_reply,
)
from recipes.beta.spade.designer_processor import SpadeDesignerProcessor
from recipes.beta.spade.generation import (
    Checks,
    Designer,
    Generation,
    GenerationError,
    GenerationRequest,
    GenerationResult,
    RealChecks,
    ReasoningAgent,
    ReefDesigner,
    ReefReasoningAgent,
    load_experience,
)
from recipes.beta.spade.harbor import (
    GeneratedHarborTask,
    OracleResult,
    content_hash,
    harbor_task,
    oracle_check,
    reply_errors,
    split_generation,
)
from recipes.beta.spade.harness import SpadeDesignerHarnessRecipe
from recipes.beta.spade.preparer import SpadePreparer
from recipes.beta.spade.processor import SpadeProcessor
from recipes.beta.spade.recipe import SpadeDesignerRecipe, SpadeRecipe
from recipes.beta.spade.roles import HarborAgent, Role, RoleVersion
from recipes.beta.spade.rounds import Deployment, RoundResult, SpadeRun

__all__ = [
    "Checks",
    "Deployment",
    "Designer",
    "DesignerReplyError",
    "DesignerRequest",
    "GeneratedHarborTask",
    "Generation",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "HarborAgent",
    "HarborReply",
    "OracleResult",
    "PlayRecord",
    "RealChecks",
    "ReasoningAgent",
    "ReefDesigner",
    "ReefReasoningAgent",
    "Role",
    "RoleVersion",
    "RoundResult",
    "SpadeDesignerHarnessRecipe",
    "SpadeDesignerProcessor",
    "SpadeDesignerRecipe",
    "SpadePreparer",
    "SpadeProcessor",
    "SpadeRecipe",
    "SpadeRun",
    "content_hash",
    "designer_messages",
    "designer_prompt",
    "harbor_task",
    "load_experience",
    "oracle_check",
    "parse_harbor_reply",
    "reply_errors",
    "split_generation",
]
