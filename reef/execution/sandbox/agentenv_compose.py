"""Compatibility path for AgentENV Compose helpers."""

import sys

from reef.execution.sandbox.agentenv import compose

COMPOSE_FILE = compose.COMPOSE_FILE
COMPOSE_MODES = compose.COMPOSE_MODES
COMPOSE_VARIABLE = compose.COMPOSE_VARIABLE
GUEST_BASH_COMMAND = compose.GUEST_BASH_COMMAND
IMAGE_NOT_FOUND_TEXT = compose.IMAGE_NOT_FOUND_TEXT
MAIN_COMMAND = compose.MAIN_COMMAND
PROJECT_NAME = compose.PROJECT_NAME
AgentEnvComposeOps = compose.AgentEnvComposeOps
ComposeHost = compose.ComposeHost
ComposeSandbox = compose.ComposeSandbox
ComposeStartError = compose.ComposeStartError
ImageNotFoundError = compose.ImageNotFoundError
TransientStartError = compose.TransientStartError
aenv_environment = compose.aenv_environment
compose_mode = compose.compose_mode
compose_project = compose.compose_project
delete_sandbox = compose.delete_sandbox
main_image = compose.main_image
prepare_guest = compose.prepare_guest
run_aenv = compose.run_aenv

# Both import paths must see changes to module-level settings and hooks.
sys.modules[__name__] = compose
