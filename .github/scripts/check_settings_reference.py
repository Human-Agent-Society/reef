#!/usr/bin/env python3
"""Keep ``docs/reference/configuration-settings.rst`` equal to the settings that the code declares.

The page lists every setting of a deployment config: its key, type, default and the ``help`` text of its
declaration. This script writes the page from the declarations, so the page cannot differ from the code.

Run ``python .github/scripts/check_settings_reference.py --write`` after you add or change a setting.
Without ``--write``, the script fails when the page is stale, or when a settings class under ``reef`` is
neither on the page nor in ``EXCLUDED``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import pkgutil
import re
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import reef
from reef.api.settings import service_config_arguments
from reef.artifact.git_lfs import GitLFSSettings
from reef.artifact.sources import S3ReaderSettings
from reef.deploy.deployment_config import deployment_config_arguments, public_name, translate_layout
from reef.deploy.generator import GeneratorSettings
from reef.evaluation.harness.settings import EvaluationSettings
from reef.execution.executor.config import ExecutorSettings, WorkerResources
from reef.inference.http import InferenceProxySettings
from reef.inference.remote.config import RemoteInferenceSettings
from reef.inference.tinker import TinkerInferenceSettings
from reef.observability.tracing import TracingSettings
from reef.observability.wandb import WandbSettings
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.generated_tasks.processor import GeneratedTaskSettings
from reef.recipe.sft.processor import SFTSettings
from reef.routing.settings import RoutingDeploymentSettings
from reef.samples.distill import DistillSamplesSettings
from reef.samples.trajectories import TrajectorySettings
from reef.storage.sql.data import PostgresDataStoreSettings
from reef.storage.sql.postgres import PostgresStorageSettings
from reef.train.harness_backend.agent.method import AgentMethodSettings
from reef.train.harness_backend.settings import EvolutionSettings
from reef.train.sample2job.base import Sample2JobSettings
from reef.train.sample2job.traces import TraceJobDataSettings
from reef.train.tinker_backend.config import SpindleFullSettings, TinkerServiceSettings, TinkerSettings

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "reference" / "configuration-settings.rst"
COMMAND = "python .github/scripts/check_settings_reference.py --write"

TYPE_NAMES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "strings": "list",
    "object": "object",
    "plugin": "plugin",
    "plugins": "plugin list",
}


@dataclass(frozen=True)
class Table:
    """One block of rows: the settings of ``settings_type`` under the config path ``prefix``.

    ``condition`` is the sentence that says when the keys apply; empty for keys that always apply.
    With ``own_fields_only``, the block leaves out the fields that ``settings_type`` inherits, because
    another block lists them. ``omitted`` names the fields that a config cannot set, each with the reason.
    ``undeclared`` gives the help of a field that the class reads by hand and so declares without one.
    With ``underscores``, the keys keep the underscores of the field names, for a section that refuses hyphens.
    """

    prefix: str
    settings_type: type
    condition: str = ""
    own_fields_only: bool = False
    omitted: Mapping[str, str] = dataclasses.field(default_factory=dict)
    undeclared: Mapping[str, str] = dataclasses.field(default_factory=dict)
    underscores: bool = False


@dataclass(frozen=True)
class Section:
    """One section of the page: the top-level keys whose first segment is in ``names``, then ``tables``."""

    title: str
    names: tuple[str, ...]
    intro: str
    tables: tuple[Table, ...] = ()


MANAGED_TIMEOUTS = {
    "inference_timeout_s": "Reef fills it from inference.timeout-s",
    "train_timeout_s": "Reef fills it from training.timeout-s",
}

SECTIONS = (
    Section(
        "Service",
        ("reef", "harness"),
        "The HTTP service, its tokens and its scenario limits.",
        (
            Table(
                "reef.routing-deployment",
                RoutingDeploymentSettings,
                "The Semantic Router of a deployment with ``reef.routing-manifest``.",
                undeclared={
                    "restart_backoff_seconds": (
                        "Seconds to wait before each recovery attempt, as a list. Later attempts reuse the last value."
                    )
                },
            ),
        ),
    ),
    Section(
        "Storage",
        ("storage", "data"),
        "The stores of a deployment. A backend key takes a built-in name, or a plugin with ``module`` and ``config``.",
        (
            Table("storage.artifact-backend.config", GitLFSSettings, "With ``module: git_lfs``."),
            Table("storage.record-backend.config", PostgresStorageSettings, "With ``module: postgres``."),
            Table("storage.data-store.config", PostgresDataStoreSettings, "With ``module: postgres``."),
            Table(
                "storage.artifact-sources[].config",
                S3ReaderSettings,
                "With ``module: reef.artifact.sources:S3Reader``.",
            ),
        ),
    ),
    Section(
        "Inference",
        ("inference",),
        "The model or provider that answers requests, and the serving engines of a weight-training deployment.",
        (
            Table(
                "inference.remote",
                RemoteInferenceSettings,
                "The remote inference component that runs the engines.",
            ),
        ),
    ),
    Section(
        "Training",
        ("training", "teacher"),
        "The training backend of a weight-training deployment, and the teacher engine.",
        (
            Table(
                "training.options",
                TinkerServiceSettings,
                "With ``training.backend: tinker`` or ``spindle_full``. Slime takes its own native flags here.",
                omitted=MANAGED_TIMEOUTS,
            ),
            Table("training.options", TinkerSettings, "With ``training.backend: tinker``.", own_fields_only=True),
            Table(
                "training.options",
                SpindleFullSettings,
                "With ``training.backend: spindle_full``.",
                own_fields_only=True,
            ),
        ),
    ),
    Section(
        "Recipe",
        ("recipe", "activation", "pipeline"),
        "The recipe of the deployment, the rule that decides which artifacts serve, and the pipeline stages.",
        (
            Table("recipe.config", WeightTrainingRecipe, "Every recipe has ``training-mode``."),
            Table("recipe.runtime", InferenceProxySettings, "With ``recipe.runtime.type: inference_proxy``."),
            Table("recipe.runtime", TinkerInferenceSettings, "With ``recipe.runtime.type: tinker``."),
            Table(
                "pipeline.sample2job.config",
                Sample2JobSettings,
                "Every built-in sample-to-job stage that trains on records.",
            ),
            Table(
                "pipeline.sample2job.config",
                TraceJobDataSettings,
                "The stages ``TraceJobData`` and ``TrainTaskJobData`` of harness evolution.",
                own_fields_only=True,
            ),
            Table("pipeline.record2sample.config", TrajectorySettings, "The stage ``TrajectorySamples``."),
            Table("pipeline.record2sample.config", DistillSamplesSettings, "The stage ``DistillSamples``."),
            Table("pipeline.record2sample.config", SFTSettings, "The stage ``SFTSamples``."),
            Table("pipeline.record2sample.config", GeneratedTaskSettings, "The stage ``GeneratedTaskSamples``."),
        ),
    ),
    Section(
        "Harness evolution",
        (),
        "The keys of ``recipe.config.evolution`` for ``reef.recipe.harness:HarnessEvolutionRecipe``.",
        (
            Table("recipe.config.evolution", EvolutionSettings),
            Table("recipe.config.evolution.evaluation", EvaluationSettings),
            Table(
                "recipe.config.evolution.method.config",
                AgentMethodSettings,
                "With ``method.module: reef.train.harness_backend.agent:AgentMethod``.",
            ),
        ),
    ),
    Section(
        "Evaluation",
        ("evaluation", "task-import"),
        "The evaluation service, and the Harbor tasks that each scenario imports.",
    ),
    Section(
        "Generator",
        ("generator",),
        "The generator service, which writes and checks Harbor tasks and runs their rollouts.",
        (Table("generator", GeneratorSettings),),
    ),
    Section(
        "Executors",
        (),
        "``execution.<role>`` selects how Reef starts the workers of a role: ``training`` (Slime's training "
        "workers), ``rollout`` (the serving engines), ``generator`` (the generator service) or ``evolution`` "
        "(harness evaluation workers). A selector is a backend (a built-in name or ``package.module:ClassName``), "
        "the name of a profile under ``executors``, or an object with these keys. A profile takes the same keys.",
        (
            Table("execution.<role>", ExecutorSettings),
            Table("execution.<role>.resources", WorkerResources),
        ),
    ),
    Section(
        "Observability",
        ("observability",),
        "Experiment tracking and record tracing.",
        (
            Table("observability.wandb", WandbSettings, "These keys take underscores only.", underscores=True),
            Table("observability.tracing", TracingSettings, "These keys take underscores only.", underscores=True),
        ),
    ),
)

#: Top-level keys whose default the deploy layer takes from another key. The declared default applies only to
#: a Python caller that builds the settings itself, so the page names the other key.
DERIVED_DEFAULTS = {"inference.retry-timeout-s": "``inference.timeout-s``"}

#: Settings classes that the page does not list, each with the reason.
EXCLUDED = {
    "reef.api.settings.ServiceSettings": "its fields are the top-level keys, listed by their public paths",
    "reef.deploy.deployment_config.DeploymentSettings": "its fields are top-level keys of the reef section",
    "reef.recipe.base.Recipe": "listed with WeightTrainingRecipe, which inherits its fields",
    "reef.samples.assembly.SampleAssemblySettings": "the base of DistillSamplesSettings",
    "reef.inference.remote.runtime.RemoteServingRuntimeSettings": "Reef builds this config from the inference section",
    "reef.tinker.weights.TinkerSamplerWeightsSettings": "the Tinker backend builds this config",
}


class SettingsReferenceError(RuntimeError):
    """The page is stale, or a settings class is neither listed nor excluded."""


def default_text(default: object) -> str:
    if default is dataclasses.MISSING:
        return "required"
    if default is None:
        return "unset"
    if isinstance(default, bool):
        return "true" if default else "false"
    if isinstance(default, float) and default.is_integer():
        return str(int(default))
    if isinstance(default, str):
        return default if default else "empty"
    if isinstance(default, (tuple, list, dict)):
        return json.dumps(default)
    return str(default)


def escaped(text: str) -> str:
    """Help text is plain text: stop reStructuredText from reading markup in it."""
    text = re.sub(r"([\\*`|])", r"\\\1", text)
    return re.sub(r"(?<=\w)_(?=\W|$)", r"\\_", text)


def row(key: str, type_name: str, default: object, help_text: str, env: str | None = None) -> list[str]:
    """One row of a config block; ``default`` is ``dataclasses.MISSING`` for a required setting."""
    description = escaped(" ".join(help_text.split()))
    if not description:
        raise SettingsReferenceError(f"{key} has no help text; give its declaration one")
    if env:
        description += f" Environment fallback: ``{env}``."
    return [f"   ``{key}`` : {type_name} : {default_text(default)}", f"       {description}", ""]


def type_name(annotation: object) -> str:
    """The type of a setting as the page names it; the same names as ``TYPE_NAMES`` gives a declared kind."""
    members = [member for member in get_args(annotation) if member is not type(None)]
    if get_origin(annotation) in (Union, types.UnionType):
        if members == [str, Mapping[str, Any]]:
            return "plugin"
        return type_name(members[0])
    origin = get_origin(annotation) or annotation
    if origin is Literal:
        return type_name(type(members[0]))
    if origin in (tuple, list):
        return "plugin list" if members and get_origin(members[0]) in (Union, types.UnionType) else "list"
    if origin in (Mapping, dict):
        return "object"
    names = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if origin not in names:
        raise SettingsReferenceError(f"no type name for the annotation {annotation!r}")
    return names[origin]


def table_rows(table: Table) -> list[str]:
    """The rows of the fields of ``table.settings_type`` that a config can set, in declaration order.

    A field belongs when it declares a setting or ``table.undeclared`` names it.
    """
    hints = get_type_hints(table.settings_type)
    own = vars(table.settings_type).get("__annotations__", {})
    fields = {field.name: field for field in dataclasses.fields(table.settings_type)}
    unknown = (set(table.omitted) | set(table.undeclared)) - set(fields)
    if unknown:
        raise SettingsReferenceError(
            f"{table.settings_type.__name__} has no field {', '.join(sorted(unknown))}; correct the Table"
        )
    rows = []
    for field in fields.values():
        declared = "config_help" in field.metadata
        if not (declared or field.name in table.undeclared) or field.name in table.omitted:
            continue
        if table.own_fields_only and field.name not in own:
            continue
        default = field.default
        if default is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        rows += row(
            f"{table.prefix}.{field.name if table.underscores else field.name.replace('_', '-')}",
            type_name(hints[field.name]),
            default,
            field.metadata["config_help"] if declared else table.undeclared[field.name],
            field.metadata.get("config_env"),
        )
    return rows


def lookup(config: object, path: tuple[str, ...]) -> object:
    node = config
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return dataclasses.MISSING
        node = node[part]
    return node


def top_level_rows() -> dict[str, list[str]]:
    """The rows of the top-level keys, by first segment, in declaration order."""
    seeds = translate_layout({})
    rows: dict[str, list[str]] = {}
    for argument in (*service_config_arguments(), *deployment_config_arguments()):
        key = public_name(argument)
        default = lookup(seeds, argument.path)
        if default is dataclasses.MISSING and not argument.required:
            default = argument.default
        default = DERIVED_DEFAULTS.get(key, default)
        rows.setdefault(key.split(".")[0], []).extend(
            row(key, TYPE_NAMES[argument.kind], default, argument.help, argument.env)
        )
    return rows


def render() -> str:
    top_level = top_level_rows()
    unlisted = set(top_level) - {name for section in SECTIONS for name in section.names}
    if unlisted:
        raise SettingsReferenceError(f"no section lists the top-level keys of: {', '.join(sorted(unlisted))}")
    lines = [
        ".. This page is generated. Do not edit it. Change the ``help`` of the setting, then run:",
        f"   {COMMAND}",
        "",
        "Settings reference",
        "==================",
        "",
        "Every setting of a deployment config, with its type, its default and what it does. Each key is also a "
        "command-line flag of ``reef serve`` with the same path, such as ``--inference.model-path``. "
        "`Configure a deployment <configuration.rst>`__ gives the layout of the file and the value rules.",
        "",
        "A key can use hyphens or underscores, except where a table says otherwise. ``unset`` is a key without "
        "a value: Reef then uses the behavior that the description gives. An ``object`` holds keys that another "
        "table lists, or the options of a backend. A ``plugin`` is a built-in name, a "
        "``package.module:ClassName`` reference, or an object with ``module`` and ``config``.",
        "",
    ]
    for section in SECTIONS:
        lines += [section.title, "-" * len(section.title), "", section.intro, ""]
        section_rows = [line for name in section.names for line in top_level.get(name, [])]
        if section_rows:
            lines += [".. container:: config", "", *section_rows]
        for table in section.tables:
            label = f"**{table.prefix}.** {table.condition}".rstrip()
            lines += [label, "", ".. container:: config", "", *table_rows(table)]
    return "\n".join(lines).rstrip() + "\n"


def declared_settings_classes() -> set[str]:
    """Every dataclass under ``reef`` that declares a setting itself; a module that cannot import is skipped."""
    found = set()
    for info in pkgutil.walk_packages(reef.__path__, "reef."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue
        for name, value in vars(module).items():
            if not isinstance(value, type) or not dataclasses.is_dataclass(value) or value.__module__ != info.name:
                continue
            own = vars(value).get("__annotations__", {})
            if any("config_help" in field.metadata and field.name in own for field in dataclasses.fields(value)):
                found.add(f"{info.name}.{name}")
    return found


def check_coverage() -> None:
    listed = set()
    for section in SECTIONS:
        for table in section.tables:
            listed.add(f"{table.settings_type.__module__}.{table.settings_type.__name__}")
    missing = declared_settings_classes() - listed - set(EXCLUDED)
    if missing:
        raise SettingsReferenceError(
            "these settings classes are not on the settings reference page; add a Table to SECTIONS, "
            f"or a reason to EXCLUDED: {', '.join(sorted(missing))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the page instead of checking it.")
    arguments = parser.parse_args()
    try:
        check_coverage()
        text = render()
        if arguments.write:
            PAGE.write_text(text, encoding="utf-8")
        elif not PAGE.is_file() or PAGE.read_text(encoding="utf-8") != text:
            raise SettingsReferenceError(f"{PAGE.relative_to(ROOT)} is stale; run: {COMMAND}")
    except SettingsReferenceError as error:
        print(f"settings reference: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
