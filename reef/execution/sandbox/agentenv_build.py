"""Compatibility path for AgentENV template build helpers."""

import sys

from reef.execution.sandbox.providers.agentenv import build

ALIAS_CONFLICT_TEXT = build.ALIAS_CONFLICT_TEXT
BUILD_OUTPUT_TAIL_CHARS = build.BUILD_OUTPUT_TAIL_CHARS
BUILD_POLL_INTERVAL_S = build.BUILD_POLL_INTERVAL_S
BUILD_STATUSES = build.BUILD_STATUSES
CREATED_TEMPLATE = build.CREATED_TEMPLATE
DOCKERFILE_BUILDS = build.DOCKERFILE_BUILDS
DOCKERFILE_BUILD_VARIABLE = build.DOCKERFILE_BUILD_VARIABLE
ORPHAN_BUILDING_GRACE_S = build.ORPHAN_BUILDING_GRACE_S
ORPHAN_WAITING_S = build.ORPHAN_WAITING_S
PROXY_ERROR_STATUS = build.PROXY_ERROR_STATUS
TRANSIENT_STATUSES = build.TRANSIENT_STATUSES
AgentEnvBuildError = build.AgentEnvBuildError
AgentEnvTemplates = build.AgentEnvTemplates
AliasConflictError = build.AliasConflictError
DockerfileBuild = build.DockerfileBuild
TemplateState = build.TemplateState
TransientBuildError = build.TransientBuildError
build_image = build.build_image
delete_orphan = build.delete_orphan
dockerfile_build = build.dockerfile_build
dockerfile_sets_start_command = build.dockerfile_sets_start_command
ensure_aenv_template = build.ensure_aenv_template
is_transient = build.is_transient
poll_interval = build.poll_interval

# Both import paths must see changes to module-level settings and hooks.
sys.modules[__name__] = build
