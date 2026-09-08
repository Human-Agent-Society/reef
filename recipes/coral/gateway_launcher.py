"""Wire the adapter into a CORAL gateway (optional CORAL dependency).

CORAL's ``GatewayManager.start()`` builds ``CoralGatewayMiddleware`` around
LiteLLM's app and hands the result to uvicorn. This module splices the reef
layer into that middleware's inner ``app`` reference:

    uvicorn -> CoralGatewayMiddleware -> ReefGatewayMiddleware -> LiteLLM

CORAL stamps ``x-coral-agent-id``/``x-coral-session-id`` first, then the
reef layer translates them to reef headers, so the adapter never needs
CORAL's key registry. The outer middleware object is left in place —
``GatewayManager.register_agent`` type-checks it, so replacing it would
break agent registration. Requires only that reef is one of the LiteLLM
upstreams (an ``api_base`` pointing at ``reef serve``).

Written against CORAL commit a69cbc2; the touched surface is the documented
middleware's ``app`` attribute only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recipes.coral.journal import CallJournal
from recipes.coral.middleware import ReefGatewayMiddleware


def attach_reef_adapter(
    gateway_manager: Any,
    *,
    scenario: str,
    journal_path: Path,
    extra_tags: Mapping[str, str] | None = None,
) -> CallJournal:
    """Arrange for the reef layer to sit under a CORAL gateway.

    Call between ``GatewayManager(...)`` construction and ``start()``: it
    wraps the manager's ``start`` so the reef layer is spliced in right
    after CORAL builds its middleware. Returns the journal for the
    reporter side.
    """
    journal = CallJournal(journal_path)
    original_start = gateway_manager.start

    def start_with_adapter() -> None:
        original_start()
        insert_reef_layer(
            gateway_manager._middleware,
            scenario=scenario,
            journal=journal,
            extra_tags=extra_tags,
        )

    gateway_manager.start = start_with_adapter
    return journal


def insert_reef_layer(
    coral_middleware: Any,
    *,
    scenario: str,
    journal: CallJournal,
    extra_tags: Mapping[str, str] | None = None,
) -> None:
    """Splice :class:`ReefGatewayMiddleware` under an existing CORAL middleware.

    Idempotent: a middleware whose ``app`` is already the reef layer is left
    alone, so a retried launcher does not stack two layers.
    """
    if coral_middleware is None or not hasattr(coral_middleware, "app"):
        raise TypeError("expected a started CoralGatewayMiddleware with an `app` attribute")
    if isinstance(coral_middleware.app, ReefGatewayMiddleware):
        return
    coral_middleware.app = ReefGatewayMiddleware(
        coral_middleware.app,
        scenario=scenario,
        journal=journal,
        extra_tags=extra_tags,
    )
