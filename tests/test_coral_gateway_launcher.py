"""Launcher splice: the reef layer goes under CORAL's middleware, not around it."""

from __future__ import annotations

import pytest

from recipes.beta.coral.gateway_launcher import (
    attach_reef_adapter,
    attach_reef_adapter_to_agent_manager,
    insert_reef_layer,
)
from recipes.beta.coral.middleware import ReefGatewayMiddleware


class FakeCoralMiddleware:
    """Shape-compatible stand-in: has .app and a register_agent contract."""

    def __init__(self, app):
        self.app = app
        self.registered = []

    def register_agent(self, agent_id, worktree_path, proxy_key):
        self.registered.append((agent_id, str(worktree_path), proxy_key))


class FakeManager:
    def __init__(self):
        self._middleware = None
        self.started = False

    def start(self):
        self._middleware = FakeCoralMiddleware(app=object())
        self.started = True

    def register_agent(self, agent_id, worktree_path):
        if not isinstance(self._middleware, FakeCoralMiddleware):
            raise RuntimeError("gateway middleware is not initialized")
        self._middleware.register_agent(agent_id, worktree_path, "sk-key")
        return "sk-key"


def test_attach_splices_under_coral_and_keeps_register_agent_working(tmp_path):
    manager = FakeManager()
    journal = attach_reef_adapter(manager, scenario="s", journal_path=tmp_path / "j.jsonl")
    manager.start()

    assert manager.started
    # the outer object is still CORAL's middleware (type checks keep passing)
    assert isinstance(manager._middleware, FakeCoralMiddleware)
    # the reef layer sits underneath
    assert isinstance(manager._middleware.app, ReefGatewayMiddleware)
    assert manager._middleware.app.journal is journal
    # and registration is untouched
    assert manager.register_agent("agent-1", tmp_path) == "sk-key"
    assert manager._middleware.registered[0][0] == "agent-1"


def test_insert_is_idempotent(tmp_path):
    from recipes.beta.coral.journal import CallJournal

    journal = CallJournal(tmp_path / "j.jsonl")
    middleware = FakeCoralMiddleware(app=object())
    insert_reef_layer(middleware, scenario="s", journal=journal)
    first = middleware.app
    insert_reef_layer(middleware, scenario="s", journal=journal)
    assert middleware.app is first


def test_insert_requires_a_started_middleware(tmp_path):
    from recipes.beta.coral.journal import CallJournal

    journal = CallJournal(tmp_path / "j.jsonl")
    with pytest.raises(TypeError, match="started CoralGatewayMiddleware"):
        insert_reef_layer(None, scenario="s", journal=journal)


class FakeAgentManager:
    """Shape-compatible stand-in for CORAL's AgentManager gateway-start step."""

    def __init__(self, gateway_enabled=True):
        self._gateway = None
        self._gateway_enabled = gateway_enabled
        self.agents_started = False

    def _start_gateway_if_enabled(self):
        if self._gateway_enabled:
            self._gateway = FakeManager()
            self._gateway.start()

    def start_all(self):
        self._start_gateway_if_enabled()
        # agents spawn after the gateway; the splice must already be in place
        self.agents_started = True


def test_attach_to_agent_manager_splices_before_agents_spawn(tmp_path):
    manager = FakeAgentManager()
    journal = attach_reef_adapter_to_agent_manager(manager, scenario="s", journal_path=tmp_path / "j.jsonl")
    manager.start_all()

    assert manager.agents_started
    assert isinstance(manager._gateway._middleware, FakeCoralMiddleware)
    assert isinstance(manager._gateway._middleware.app, ReefGatewayMiddleware)
    assert manager._gateway._middleware.app.journal is journal


def test_attach_to_agent_manager_fails_loudly_when_gateway_disabled(tmp_path):
    manager = FakeAgentManager(gateway_enabled=False)
    attach_reef_adapter_to_agent_manager(manager, scenario="s", journal_path=tmp_path / "j.jsonl")
    with pytest.raises(RuntimeError, match=r"agents\.gateway\.enabled"):
        manager.start_all()
