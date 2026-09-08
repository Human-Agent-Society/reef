"""Launcher splice: the reef layer goes under CORAL's middleware, not around it."""

from __future__ import annotations

import pytest

from recipes.coral.gateway_launcher import attach_reef_adapter, insert_reef_layer
from recipes.coral.middleware import ReefGatewayMiddleware


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
    from recipes.coral.journal import CallJournal

    journal = CallJournal(tmp_path / "j.jsonl")
    middleware = FakeCoralMiddleware(app=object())
    insert_reef_layer(middleware, scenario="s", journal=journal)
    first = middleware.app
    insert_reef_layer(middleware, scenario="s", journal=journal)
    assert middleware.app is first


def test_insert_requires_a_started_middleware(tmp_path):
    from recipes.coral.journal import CallJournal

    journal = CallJournal(tmp_path / "j.jsonl")
    with pytest.raises(TypeError, match="started CoralGatewayMiddleware"):
        insert_reef_layer(None, scenario="s", journal=journal)
