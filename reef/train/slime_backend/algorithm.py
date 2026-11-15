"""Common contract for algorithms exposed by the Slime training backend.

Subclass :class:`SlimeAlgorithm`, set the class attributes, implement the
abstract method, and decorate the class with ``@register_loss_family``
(``reef.train.slime_backend.loss_families``) so importing its module registers it::

    @register_loss_family
    class MyAlgo(SlimeAlgorithm):
        loss_family = "my-algo"
        loss_type = "custom_loss"

        def validate_specific_args(self, args, source):
            ...

Pipeline stages (each a method on the class; defaults are no-ops or shared
implementations — override only what your algorithm needs):

    1. **configure**  — ``parse_specific_options`` / ``apply_driver_options`` /
       ``validate_specific_args``: CLI parsing and backend-arg validation.
    2. **shape row**   — ``shape_sample_row``: per-sample wire-row serialization.
    3. **build batch** — ``build_rollout_data``: wire rows → Slime batch dict.
    4. **prepare**     — ``prepare_rollout``: pre-training work (teacher scoring,
       advantage hooks, etc.) in the training component.
    5. **compute loss** — registered via ``@worker_hook`` in ``objective.py``
       (worker-side, resolved by ``loss_families.resolve_objective_paths``). Two lanes:
       a family either replaces Slime's whole loss (``loss_type =
       "custom_loss"`` + a ``custom_loss_function_path`` hook) or keeps
       Slime's stock ``policy_loss`` and swaps only the per-token pg
       primitive (``uses_pg_loss_primitive = True`` + a
       ``custom_pg_loss_function_path`` hook; the adapter layer routes
       Slime's CISPO callsite onto the registered primitive). Two further
       worker lifecycle channels exist: ``reef_actor_init_hook_path`` runs
       once at actor init and ``reef_actor_pre_train_hook_path`` before each
       train step, for families that keep state on the actor (a frozen
       teacher, for one).
    6. **train step**  — ``train``: critic→actor orchestration in the training
       component; ``rollout_metrics`` for telemetry.

Stages 1-4 and 6 run in the training component's process (the HTTP service).
Stage 5 runs in Megatron workers that only receive a serialized ``args``
Namespace. The component stamps
``args.loss_family``; the worker calls ``loss_families.resolve_objective_paths`` during
init to import the algorithm's ``objective.py`` (triggering ``@worker_hook``
decorators) and project the dotted paths onto ``args``.
``bind()`` returns a runtime-bound instance (``self`` for stateless algorithms;
override to carry per-run state such as a teacher client or a train schedule).

Adding a family
---------------

A family lives in its method package beside its objective hooks:

    my_method/slime/
      __init__.py     the spec: a SlimeAlgorithm subclass, no torch; when the
                      family's wire row is not the default five columns, its
                      row shaper and payload builder live here too
      objective.py    the @worker_hook hooks, torch (worker side)
      <concern>.py    further worker-side torch modules, imported only from
                      objective.py's hooks (e.g. a Megatron teacher)

Register each declared hook with ``@worker_hook`` so the worker resolves and
validates it at init time::

    @worker_hook("custom_loss_function_path")
    def my_loss(args, batch, logits, sum_of_sample_mean): ...

The package ``__init__`` registers only the method's objective, by reference —
``register_objective("<name>", "my_method.slime:MyMethodObjective")``
(:mod:`reef.train.slime_backend.objectives`) — so the package's public half never imports
the Slime side. The first resolution of the objective imports ``my_method.slime``, and
``@register_loss_family`` there registers the family. Naming conventions, enforced by
``tests/slime_backend/test_slime_algorithm_contract.py``: the spec class is
``<Pkg>Algorithm`` and its driver options ``<Pkg>Settings``, both importable
from the family package root; ``@worker_hook`` entry points are ``<pkg>_loss``,
``<pkg>_advantages``, ``<pkg>_actor_init`` / ``<pkg>_actor_pre_train``; family
CLI flags are ``--<pkg>-*``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

from reef.core.job_data import TrajectorySample
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.policy_rows import build_policy_rollout_data

#: ``none`` is ``f(r) = 1``; ``truncated`` is ``min(r, upper)``; ``masked`` is
#: ``r`` strictly inside ``(lower, upper)`` and 0 outside.
POLICY_GRADIENT_WEIGHT_KINDS = ("none", "truncated", "masked")


def is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class PolicyGradientWeight:
    """The weight ``f(r)``, ``r = p / q``, a family's loss puts on the sampled token's score.

    A loss of the form ``-A_t * sg[f(p_t / q_t)] * log p_t``, with ``q`` the
    rollout engine's probability, declares its ``f`` with this value through
    :meth:`SlimeAlgorithm.policy_gradient_weight`: plain off-policy
    REINFORCE is ``none``, truncated importance sampling ``truncated`` with
    its cap as ``upper``, and masked importance sampling ``masked`` with its
    trust region as ``(lower, upper)``. Score centering reads it to add the
    matching correction term.
    """

    kind: str
    lower: float = 0.0
    upper: float = math.inf

    def __post_init__(self) -> None:
        if self.kind not in POLICY_GRADIENT_WEIGHT_KINDS:
            raise ValueError(f"policy-gradient weight kind must be one of: {', '.join(POLICY_GRADIENT_WEIGHT_KINDS)}")
        if self.kind == "truncated" and (not is_finite_number(self.upper) or self.upper <= 0):
            raise ValueError("a truncated policy-gradient weight needs a finite cap upper > 0")
        if self.kind == "masked" and (
            not is_finite_number(self.lower) or not is_finite_number(self.upper) or not 0 <= self.lower < self.upper
        ):
            raise ValueError("a masked policy-gradient weight needs finite bounds 0 <= lower < upper")


@dataclass(frozen=True)
class TrainResult:
    """One job's training: the actor workers' results and the method's durable metrics.

    ``actor_trained`` is false when the job trained only the critic (a warmup
    step): the actor took no optimizer step, so its learning-rate schedule
    does not advance.
    """

    worker_results: list[Any]
    durable_metrics: dict[str, Any] = field(default_factory=dict)
    actor_trained: bool = True


# --- worker hook registration (worker-side) ---

WORKER_HOOKS: dict[str, dict[str, str]] = {}
"""Maps ``objective.py`` module paths to ``{arg_name: fn_name}`` dicts.

Populated at import time by the :func:`worker_hook` decorator.  The worker
calls ``loss_families.resolve_objective_paths`` to project these onto ``args``.
"""


def worker_hook(arg_name: str) -> Callable[[Callable], Callable]:
    """Mark a function as the hook that a Slime worker calls through the argument ``arg_name``.

    Records the function's dotted path so the worker can resolve it via
    ``loss_families.resolve_objective_paths`` without the driver ever importing the
    (torch-dependent) ``objective.py`` module.
    """

    def decorator(fn: Callable) -> Callable:
        WORKER_HOOKS.setdefault(fn.__module__, {})[arg_name] = fn.__name__
        return fn

    return decorator


class SlimeAlgorithm(ABC):
    """Base class for Slime training algorithms.

    Required (subclass must set):
        ``loss_family``      — canonical family name used on the wire.
        ``loss_type``        — Slime ``--loss-type`` value.
        ``validate_specific_args`` — validate algorithm-specific backend args.

    Optional (sensible defaults provided):
        ``build_rollout_data``        — custom payload builder.
        ``shape_sample_row``          — custom wire-row shaper.
        ``parse_specific_options``    — parse family-specific CLI flags.
        ``apply_driver_options``      — apply parsed options to Slime's Namespace.
        ``configure_backend_args``    — project family settings onto backend args.
        ``bind``                      — return a runtime-bound instance.
        ``prepare_rollout``           — pre-training work in the training component.
        ``train``                     — critic→actor orchestration.
        ``rollout_metrics``           — telemetry after training.
        ``policy_gradient_weight``    — the loss's per-token weight, for score centering.
    """

    # --- required class attributes (set in subclass) ---
    loss_family: str
    loss_type: str

    # --- optional class attributes (defaults shown) ---
    #: Whether the family trains on rows whose rollout log-probabilities come from Slime's own engines, as dataset
    #: jobs do; every other family trains rows without them, so it refuses ``--use-rollout-logprobs``.
    accepts_rollout_logprobs: bool = False
    advantages: Literal["required", "forbidden", "optional"] = "optional"
    forbidden_advantages_message: str | None = None
    allows_slime_advantage_computation: bool = False
    requires_driver_options: bool = False
    missing_driver_options_message: str | None = None

    # --- optional wire-surface declarations (consumed generically by the
    # reef_adapters layer via ``configure_reef_loss_args``) ---
    rollout_data_keys: tuple[str, ...] = ()
    """Extra per-sample payload keys the training component partitions across DP ranks."""
    rollout_tensor_dtypes: Mapping[str, str] = {}
    """Payload keys to tensorize before training, as {key: "int"|"long"|"float32"}."""
    external_batch_keys: tuple[str, ...] = ()
    """Rollout-data keys the Megatron worker forwards through ``get_batch``."""
    rollout_log_skip_keys: tuple[str, ...] = ()
    """Non-scalar rollout-data keys hidden from Slime's numeric rollout logger."""
    response_aligned_keys: tuple[str, ...] = ()
    """Rollout-data tensors laid out per response token.

    The worker slices these for context parallelism the same way it slices
    ``advantages`` and ``returns``; each key must also appear in
    ``rollout_tensor_dtypes``.
    """
    critic_value_mask_key: str | None = None
    """Microbatch key selecting the tokens the critic's explained-variance metric covers."""
    required_objective_hooks: tuple[str, ...] = ()
    """Worker-side objective channels that must be registered during init."""
    critic_value_head_zero_init: bool = False
    """Leave the critic value head's bias at its worker-local zero initialization.

    Declared by families whose critic grows a scalar value head that has no
    counterpart tensor in the HF checkpoint; the worker's HF bootstrap then
    answers the head's bias lookup with zeros instead of a failing HF read.
    """
    uses_pg_loss_primitive: bool = False
    """Lane B loss injection: keep Slime's ``policy_loss``, swap the pg primitive.

    Declared with a ``custom_pg_loss_function_path`` ``@worker_hook`` hook. The
    adapter layer owns the routing (it points Slime's CISPO callsite at the
    registered primitive); families only declare the lane.
    """

    # --- stage 1: configure (must implement) ---

    @abstractmethod
    def validate_specific_args(self, args: Namespace, source: str) -> None:
        """Validate algorithm-specific backend args.

        ``source`` identifies the configured ``reef.recipe`` — use it in error
        messages.
        At minimum, ``pass`` if your algorithm has no specific requirements.
        """

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[Any | None, list[str]]:
        """Parse family-specific CLI flags. Default: no-op."""
        return None, list(arguments)

    def policy_gradient_weight(self, args: Namespace) -> PolicyGradientWeight | None:
        """The weight ``f(p / q)`` this family's loss puts on the sampled token's score.

        A family whose policy-gradient loss has the form
        ``-A_t * sg[f(p_t / q_t)] * log p_t``, with ``q`` the rollout engine's
        probability, returns its ``f`` so ``--score-centering`` can add the
        matching correction term to that loss. Default ``None``: the loss has
        another form (a clipped surrogate against a recomputed old policy, a
        full-distribution divergence), and score centering is refused.
        """
        return None

    def configure_backend_args(self, args: Namespace) -> None:
        """Project family-specific settings onto the parsed backend args.

        Runs after the driver stamps ``loss_family`` — the family's chance to
        derive backend settings or refuse unsupported configurations.
        Default: no-op.
        """
        return

    def apply_driver_options(self, args: Namespace, options: Any) -> None:
        """Apply parsed options to Slime's Namespace.

        Default: stamp ``loss_family`` so the worker can resolve the
        algorithm spec and register objective paths.  Subclasses that
        strip family-specific flags should call ``super()`` to preserve
        this behavior.
        """
        args.loss_family = self.loss_family

    # --- stage 2: shape row ---

    def shape_sample_row(self, sample: TrajectorySample) -> list[Any]:
        """Shape one Reef sample into this family's wire row.

        Default: ``[source_id, tokens, loss_mask, rollout_log_probs, reward]``.
        """
        return [
            source_record_id(sample),
            list(sample.training.get("tokens", [])),
            list(sample.training.get("loss_mask", [])),
            list(sample.training.get("rollout_log_probs", [])),
            trajectory_reward(sample),
        ]

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        """Build Slime rollout data from Reef wire payload.

        Default: shared policy 5-tuple builder. Override for custom wire formats.
        """
        return build_policy_rollout_data(payload, samples, self.loss_family)

    # --- stage 4: prepare rollout ---

    def validate_payload(self, rollout_data: Mapping[str, Any]) -> None:
        """Check the payload's loss field matches this family. Default: check."""
        payload_loss = rollout_data.get("loss")
        if self.loss_family and payload_loss != self.loss_family:
            raise ValueError(
                f"Slime training for loss family {self.loss_family!r} received payload loss {payload_loss!r}"
            )

    def prepare_rollout(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Pre-training work in the training component (teacher scoring, etc.). Default: no-op."""
        return {}

    def configure_critic_args(self, critic_args: Namespace) -> None:
        """Adjust the critic role's args (lambda, freeze patterns, etc.). Default: no-op."""
        return

    # --- stage 5: compute loss (worker-side, via @worker_hook decorator) ---

    # --- stage 6: train step ---

    def train(
        self,
        rollout_id: int,
        rollout_data_refs: Any,
        *,
        actor_group: Any,
        critic_group: Any,
        resolve: Callable[[Any], Any],
    ) -> TrainResult:
        """Execute one training step. Default: single actor train call."""
        results = resolve(actor_group.async_train(rollout_id, rollout_data_refs))
        return TrainResult(worker_results=list(results or ()))

    def rollout_metrics(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Return telemetry metrics after training. Default: none."""
        return {}

    # --- runtime binding ---

    def bind(
        self,
        config: object | None = None,
        *,
        critic_steps_per_actor: int | None = None,
        critic_only_steps: int = 0,
    ) -> SlimeAlgorithm:
        """Return a runtime-bound instance.

        Default: ``self`` (stateless). Override to carry per-run state such as
        a teacher client or a train schedule; call ``self.__class__()`` to
        create a fresh instance so the registry singleton stays untouched.

        The component always forwards whatever ``parse_specific_options``
        returned as ``config``, so a family whose options travel on ``args``
        via ``apply_driver_options`` must still accept its own Settings type
        here — and may ignore the value. A family whose options are consumed
        in the component stores it. ``critic_steps_per_actor`` is ``None`` when the
        operator left the flag unset; a critic family substitutes its own
        default.
        """
        if config is not None:
            raise TypeError(f"loss family {self.loss_family!r} does not accept an algorithm config")
        return self

    # --- template methods (do not override) ---

    def validate_backend_args(self, args: Namespace, *, recipe: str | None = None) -> None:
        source = f"reef.recipe={recipe}" if recipe else f"loss family {self.loss_family}"
        actual_loss = getattr(args, "loss_type", None)
        if actual_loss != self.loss_type:
            raise RuntimeError(f"{source} requires --loss-type {self.loss_type}, got {actual_loss!r}")
        if getattr(args, "use_rollout_logprobs", False) and not self.accepts_rollout_logprobs:
            # Serving returns no log-probabilities: the actor computes the old-policy log-probabilities of each row
            # with the weights of the job's start.
            raise RuntimeError(
                f"{source} trains rows without rollout log-probabilities: remove --use-rollout-logprobs, so that "
                "the actor computes the old-policy log-probabilities"
            )
        self.validate_specific_args(args, source)

    def parse_driver_options(self, arguments: Sequence[str]) -> tuple[Any | None, list[str]]:
        options, remaining = self.parse_specific_options(arguments)
        if self.requires_driver_options and options is None:
            message = self.missing_driver_options_message or (
                f"Slime loss family {self.loss_family!r} requires algorithm-specific driver options"
            )
            raise RuntimeError(message)
        return options, remaining
