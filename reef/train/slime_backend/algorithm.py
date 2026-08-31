"""Common contract for algorithms exposed by the Slime training backend.

Subclass :class:`SlimeAlgorithm`, set the class attributes, implement the
abstract method, and decorate the class with ``@register_loss_family`` so
importing the package registers it::

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
       advantage hooks, etc.) on the bridge actor.
    5. **compute loss** — registered via ``@objective`` in ``objective.py``
       (worker-side, resolved by :func:`resolve_objective_paths`). Two lanes:
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
    6. **train step**  — ``train``: critic→actor orchestration on the bridge
       actor; ``provenance_metrics`` for telemetry.

Stages 1-4 and 6 run in the driver process.  Stage 5 runs in Megatron workers
that only receive a serialized ``args`` Namespace.  The driver stamps
``args.loss_family``; the worker calls :func:`resolve_objective_paths` during
init to import the algorithm's ``objective.py`` (triggering ``@objective``
decorators) and project the dotted paths onto ``args``.
``bind()`` returns a runtime-bound instance (``self`` for stateless algorithms;
override to carry per-run state such as a teacher client or a train schedule).

Adding a family
---------------

A family lives in its method package beside its objective hooks:

    my_method/slime/
      __init__.py     the spec: a SlimeAlgorithm subclass, no torch
      objective.py    the @objective hooks, torch (worker side)
      <concern>.py    further worker-side torch modules, imported only from
                      objective.py's hooks (e.g. a Megatron teacher)
      utils/          driver-side helpers, never importing torch; only when
                      the family's wire row is not the default five columns

Register each declared hook with ``@objective`` so the worker resolves and
validates it at init time::

    @objective("custom_loss_function_path")
    def my_loss(args, batch, logits, sum_of_sample_mean): ...

Register the spec by reference from the package ``__init__`` —
``register_loss_family_ref("<name>", "my_method.slime:MyMethodAlgorithm")``
(:mod:`reef.train.algos.registry`) — so the service never imports
the Slime side and the driver imports it on first resolve. Naming conventions, enforced by
``tests/reef_service/test_slime_algorithm_contract.py``: the spec class is
``<Pkg>Algorithm`` and its driver options ``<Pkg>Settings``, both importable
from the family package root; ``@objective`` entry points are ``<pkg>_loss``,
``<pkg>_advantages``, ``<pkg>_actor_init`` / ``<pkg>_actor_pre_train``; family
CLI flags are ``--<pkg>-*``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TrainResult:
    worker_results: list[Any]
    durable_metrics: dict[str, Any] = field(default_factory=dict)


# --- objective registration (worker-side) ---

_objective_registry: dict[str, dict[str, str]] = {}
"""Maps ``objective.py`` module paths to ``{arg_name: fn_name}`` dicts.

Populated at import time by the :func:`objective` decorator.  The worker
calls :func:`resolve_objective_paths` to project these onto ``args``.
"""


def objective(arg_name: str) -> Callable[[Callable], Callable]:
    """Mark a function as a Slime objective hook for ``arg_name``.

    Records the function's dotted path so the worker can resolve it via
    :func:`resolve_objective_paths` without the driver ever importing the
    (torch-dependent) ``objective.py`` module.
    """

    def decorator(fn: Callable) -> Callable:
        _objective_registry.setdefault(fn.__module__, {})[arg_name] = fn.__name__
        return fn

    return decorator


def resolve_args_loss_family(args: Namespace) -> SlimeAlgorithm:
    """Resolve the loss family a worker's ``args`` names.

    The driver stamps the canonical ``loss_family``. For a family it reached
    through a dotted ``package.module:SPEC`` reference it also stamps that
    reference on ``loss_family_ref``: the driver's registry does not travel
    with the Namespace, so a fresh worker process has to import the module
    itself. The reference wins when present.
    """
    from reef.train.slime_backend.loss_families import resolve_loss_family

    return resolve_loss_family(getattr(args, "loss_family_ref", None) or args.loss_family)


def resolve_objective_paths(args: Namespace) -> None:
    """Resolve custom objective function paths onto ``args`` (worker-side).

    Called during worker init.  Imports the algorithm's ``objective`` module
    (triggering ``@objective`` decorators) and projects each registered
    dotted path onto ``args``.  A family without an ``objective`` module is
    skipped when it declares no hooks.  Import failures raised from
    inside an objective module propagate so a broken worker dependency cannot
    silently disable the training objective.
    """
    loss_family = getattr(args, "loss_family", None)
    if not loss_family:
        return
    spec = resolve_args_loss_family(args)
    required_hooks = set(spec.required_objective_hooks)
    if spec.loss_type == "custom_loss":
        required_hooks.add("custom_loss_function_path")
    if spec.uses_pg_loss_primitive:
        required_hooks.add("custom_pg_loss_function_path")

    module_path = f"{spec.__class__.__module__}.objective"
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name == module_path:
            if not required_hooks:
                return
            hooks = ", ".join(sorted(required_hooks))
            raise RuntimeError(
                f"loss family {loss_family!r} requires objective hooks {hooks}, but {module_path} is missing"
            ) from error
        raise

    registered = _objective_registry.get(module_path, {})
    missing_hooks = required_hooks.difference(registered)
    if missing_hooks:
        hooks = ", ".join(sorted(missing_hooks))
        raise RuntimeError(f"loss family {loss_family!r} did not register required objective hooks: {hooks}")
    for arg_name, fn_name in registered.items():
        setattr(args, arg_name, f"{module_path}.{fn_name}")


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
        ``prepare_rollout``           — pre-training work on the bridge actor.
        ``train``                     — critic→actor orchestration.
        ``provenance_metrics``        — telemetry after training.
    """

    # --- required class attributes (set in subclass) ---
    loss_family: str
    loss_type: str

    # --- optional class attributes (defaults shown) ---
    requires_rollout_logprobs: bool = False
    advantages: Literal["required", "forbidden", "optional"] = "optional"
    forbidden_advantages_message: str | None = None
    allows_slime_advantage_computation: bool = False
    requires_driver_options: bool = False
    missing_driver_options_message: str | None = None

    # --- optional wire-surface declarations (consumed generically by the
    # reef_adapters layer via ``configure_reef_loss_args``) ---
    rollout_data_keys: tuple[str, ...] = ()
    """Extra per-sample payload keys the rollout manager partitions across DP ranks."""
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

    Declared with a ``custom_pg_loss_function_path`` ``@objective`` hook. The
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

    def shape_sample_row(self, sample: Any) -> list[Any]:
        """Shape one Reef sample into this family's wire row.

        Default: ``[source_id, tokens, loss_mask, rollout_log_probs, reward]``.
        """
        return [
            sample.source_agent_record_id,
            list(sample.tokens),
            list(sample.loss_mask),
            list(sample.rollout_log_probs),
            sample.reward,
        ]

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        """Build Slime rollout data from Reef wire payload.

        Default: shared policy 5-tuple builder. Override for custom wire formats.
        """
        from reef.train.slime_backend.data_builder import build_policy_rollout_data

        return build_policy_rollout_data(payload, samples, self)

    # --- stage 4: prepare rollout ---

    def validate_payload(self, rollout_data: Mapping[str, Any]) -> None:
        """Check the payload's loss field matches this family. Default: check."""
        payload_loss = rollout_data.get("loss")
        if self.loss_family and payload_loss != self.loss_family:
            raise ValueError(
                f"Slime bridge for loss family {self.loss_family!r} received payload loss {payload_loss!r}"
            )

    def prepare_rollout(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Pre-training work on the bridge actor (teacher scoring, etc.). Default: no-op."""
        return {}

    def configure_critic_args(self, critic_args: Namespace) -> None:
        """Adjust the critic role's args (lambda, freeze patterns, etc.). Default: no-op."""
        return

    # --- stage 5: compute loss (worker-side, via @objective decorator) ---

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

    def provenance_metrics(self, rollout_data: dict[str, Any], serving_version: str) -> dict[str, Any]:
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

        The driver always forwards whatever ``parse_specific_options``
        returned as ``config``, so a family whose options travel on ``args``
        via ``apply_driver_options`` must still accept its own Settings type
        here — and may ignore the value. A family whose options are consumed
        bridge-side stores it. ``critic_steps_per_actor`` is ``None`` when the
        operator left the flag unset; a critic family substitutes its own
        default.
        """
        if config is not None:
            raise TypeError(f"loss family {self.loss_family!r} does not accept bridge algorithm config")
        return self

    # --- template methods (do not override) ---

    def validate_backend_args(self, args: Namespace, *, recipe: str | None = None) -> None:
        source = f"reef.recipe={recipe}" if recipe else f"loss family {self.loss_family}"
        actual_loss = getattr(args, "loss_type", None)
        if actual_loss != self.loss_type:
            raise RuntimeError(f"{source} requires --loss-type {self.loss_type}, got {actual_loss!r}")
        if self.requires_rollout_logprobs and not getattr(args, "use_rollout_logprobs", False):
            raise RuntimeError(
                f"{source} requires --use-rollout-logprobs so policy "
                "responses are used as the old-policy probabilities"
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


# --- loss family registration (driver-side) ---

_loss_families: dict[str, SlimeAlgorithm] = {}
"""Loss families registered at import time by :func:`register_loss_family`.

Read by :class:`~reef.train.slime_backend.loss_families.LossFamilyRegistry`
for decorated classes; instances may also be registered at runtime through
``register_loss_family(spec)``.
"""


def register_loss_family(cls: Callable[..., Any]) -> Callable[..., Any]:
    """Class decorator: instantiate and register a loss family.

    Each algorithm module applies this to its ``SlimeAlgorithm`` subclass so
    resolving the family's reference imports the module — no explicit dict entry.
    """
    instance = cls()
    name = instance.loss_family
    if name in _loss_families:
        raise ValueError(f"loss family {name!r} is already registered")
    _loss_families[name] = instance
    return cls
