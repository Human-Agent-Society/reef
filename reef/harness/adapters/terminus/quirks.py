"""Terminus adapter quirks: constructor knobs, skill frontmatter, one context module.

Terminus 2 takes its behavior from constructor arguments rather than a config
file it discovers, so ``terminus/config.json`` is a flat object whose keys must
each name a real Terminus 2 argument. An unknown key is a defect in the tree,
and failing at render is what keeps a gated change meaning what it says instead
of silently dropping a knob.

Skills carry the ``name`` and ``description`` frontmatter the instruction
builder reads, synthesized when an evolved node ships bare text, under both
skill roots. Terminus 2 has no slash-command surface, so ``agent_command``
renders under the second root and the runner names those skills as
user-invocable when it joins them.

Every model call stays on Reef's model binding. The binding writes
``model_name``, ``api_base`` and ``llm_kwargs`` with the key, renders after
the tree and wins every key it writes. The key is a credential, which a tree
cannot hold because admission refuses an inline credential, so those knobs
pass only beside it, and ``llm_kwargs`` only with the keys the binding
writes. Terminus 2 passes ``llm_call_kwargs`` to litellm on every call, over
the binding's values, so a key there that litellm reads as the endpoint, the
provider, a credential, the model, a fallback model or a logging callback
that sends the call elsewhere is refused. litellm sends a key it does not
read, and ``extra_body``, in the request body to the bound endpoint, so
neither may name a model (``model``, or OpenRouter's fallback ``models``):
the bound key only serves the bound model.

One ``code_extension`` can define an Agent subclass of Harbor's Terminus2.
Rendering only checks its syntax. Execution requires Reef's sandbox around
the runner and Harbor's remote E2B environment for the terminal task: the
task container alone does not isolate evolved Python in the outer runner.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import yaml

from reef.harness.adapters.descriptor import ExecutionValidator
from reef.harness.episodes.executor import EpisodeExecutor, EpisodeLaunchError, SandboxExecutor
from reef.harness.runners.terminus.tree import ENVIRONMENT_ENV, TerminusTreeError, extension_source
from reef.harness.tree.render import RenderError

_CONFIG = "terminus/config.json"
_SKILL_ROOTS = ("terminus/skills/", "terminus-commands/")

#: Terminus 2 constructor arguments a tree may set. Verified against harbor
#: 0.20.0, which reef-eval 0.1.1 resolves; a bump should re-check the signature.
_ALLOWED_KNOBS = {
    "enable_summarize",
    "interleaved_thinking",
    "llm_call_kwargs",
    "max_thinking_tokens",
    "max_turns",
    "parser_name",
    "proactive_summarization_threshold",
    "reasoning_effort",
    "temperature",
}
#: Set by Reef's model binding, which renders after the tree and therefore
#: wins the merge. Admitted only beside the binding's key, in llm_kwargs.
_BINDING_KNOBS = {"api_base", "llm_kwargs", "model_name"}
BINDING_LLM_KWARGS = frozenset({"api_key"})
#: litellm call arguments (read from litellm 1.102.1) that choose the endpoint, the provider, a credential, the model or a
#: fallback model, or add a logging callback that sends the call to another host. litellm sends an argument it does
#: not read, such as OpenRouter's fallback ``models``, in the request body.
MODEL_ROUTE_KWARGS = re.compile(
    r"^(?:api_base|api_key|api_version|azure|base_url|callbacks|client|client_id|client_secret"
    r"|context_window_fallback_dict|custom_llm_provider|deployment_id|failure_callback|fallbacks"
    r"|litellm_credential_name|mock_response|mock_timeout|model|model_alias_map|model_list|models|region_name"
    r"|success_callback|tenant_id|use_litellm_proxy)$"
    r"|^(?:arize|aws|azure|dd|gcs|humanloop|langfuse|langsmith|newrelic|posthog|s3|vertex|wandb|watsonx|weave)_"
    r"|^(?:adaptive|auto|complexity|quality)_router_"
)
#: Request body fields that name the model: ``model`` replaces the bound one, and OpenRouter reads ``models`` as
#: fallback models.
BODY_MODEL_KEYS = frozenset({"model", "models"})


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def _validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - _ALLOWED_KNOBS - _BINDING_KNOBS)
    if unknown:
        raise RenderError(f"terminus config sets keys that are not Terminus 2 arguments: {', '.join(unknown)}")
    llm_kwargs = config.get("llm_kwargs")
    credential = llm_kwargs.get("api_key") if isinstance(llm_kwargs, dict) else None
    bound = isinstance(credential, str) and bool(credential.strip())
    refusal = "Reef's model binding chooses the endpoint, the credential and the model"
    binding_knobs = sorted(_BINDING_KNOBS & set(config))
    if binding_knobs and not bound:
        raise RenderError(f"terminus config must not set {', '.join(binding_knobs)}: {refusal}")
    if isinstance(llm_kwargs, dict) and set(llm_kwargs) - BINDING_LLM_KWARGS:
        extra = ", ".join(sorted(set(llm_kwargs) - BINDING_LLM_KWARGS))
        raise RenderError(f"terminus config must not set llm_kwargs {extra}: {refusal}")
    call_kwargs = config.get("llm_call_kwargs")
    if call_kwargs is not None and not isinstance(call_kwargs, dict):
        raise RenderError("terminus config llm_call_kwargs must be an object")
    routed = sorted(key for key in call_kwargs or {} if MODEL_ROUTE_KWARGS.search(key))
    if routed:
        raise RenderError(f"terminus config must not set llm_call_kwargs {', '.join(routed)}: {refusal}")
    body = (call_kwargs or {}).get("extra_body")
    if body is not None and not isinstance(body, dict):
        raise RenderError("terminus config llm_call_kwargs extra_body must be an object")
    named = sorted(set(body or {}) & BODY_MODEL_KEYS)
    if named:
        raise RenderError(f"terminus config must not set llm_call_kwargs extra_body {', '.join(named)}: {refusal}")
    turns = config.get("max_turns")
    if turns is not None and (isinstance(turns, bool) or not isinstance(turns, int) or turns < 1):
        raise RenderError("terminus config max_turns must be a positive integer")


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    try:
        config = json.loads(files[_CONFIG])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RenderError("terminus primary config must be a JSON object") from exc
    if not isinstance(config, dict):
        raise RenderError("terminus primary config must be an object")
    _validate_config(config)

    try:
        extension_source(files)
    except TerminusTreeError as exc:
        raise RenderError(str(exc)) from exc

    for path, text in list(files.items()):
        if path.startswith(_SKILL_ROOTS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)

    return files


class TerminusExecutionValidator(ExecutionValidator):
    def __call__(self, files: Mapping[str, str], executor: EpisodeExecutor) -> None:
        """Refuse unisolated Python and local Docker nested in Reef's jail."""
        try:
            extension = extension_source(files)
        except TerminusTreeError as exc:
            raise EpisodeLaunchError(str(exc)) from exc
        if isinstance(executor, SandboxExecutor):
            if executor.env.get(ENVIRONMENT_ENV) != "e2b":
                raise EpisodeLaunchError(
                    "terminus Docker cannot run under evolution.executor: sandbox; "
                    "set REEF_TERMINUS_ENVIRONMENT=e2b and include it in evolution.sandbox.env_from"
                )
            if not executor.egress_hosts or not executor.env.get("E2B_API_KEY"):
                raise EpisodeLaunchError(
                    "sandboxed terminus requires egress_hosts and E2B_API_KEY in sandbox.env_from"
                )
        elif extension is not None:
            raise EpisodeLaunchError(
                "terminus code_extension requires evolution.executor: sandbox with remote E2B tasks"
            )


validate_execution = TerminusExecutionValidator()
