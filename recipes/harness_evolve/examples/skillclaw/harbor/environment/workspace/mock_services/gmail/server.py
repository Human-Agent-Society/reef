"""Mock Gmail API service for meeting negotiation agent evaluation (FastAPI on port 9100)."""

from __future__ import annotations

import json
import copy
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Gmail API")

from _base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "GMAIL_FIXTURES",
    str(Path(__file__).resolve().parent.parent
        / "fixtures" / "gmail" / "inbox.json"),
))

_emails: list[dict[str, Any]] = []
_audit_log: list[dict[str, Any]] = []
_sent_messages: list[dict[str, Any]] = []
_drafts: list[dict[str, Any]] = []
_reactive_replies: dict[str, list[dict[str, Any]]] = {}
_send_counts: dict[str, int] = {}
_triggered_replies: list[dict[str, Any]] = []


def _load_fixtures() -> None:
    global _emails, _reactive_replies, _send_counts, _triggered_replies
    global _audit_log, _sent_messages, _drafts

    _audit_log = []
    _sent_messages = []
    _drafts = []
    _send_counts = {}
    _triggered_replies = []

    with open(FIXTURES_PATH) as f:
        data = json.load(f)

    if isinstance(data, dict):
        _emails = data.get("emails", [])
        _reactive_replies = data.get("reactive_replies", {})
    else:
        _emails = data
        _reactive_replies = {}

    if not _emails:
        return

    dates = []
    for e in _emails:
        dates.append(datetime.fromisoformat(e["date"].replace("Z", "+00:00")))
    newest = max(dates)

    target = datetime.now(timezone.utc) - timedelta(days=1)
    delta = target - newest

    for e in _emails:
        old_dt = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        new_dt = old_dt + delta
        e["date"] = new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


_load_fixtures()


def _log_call(endpoint: str, request_body: dict[str, Any], response_body: Any) -> None:
    _audit_log.append({
        "endpoint": endpoint,
        "request_body": request_body,
        "response_body": response_body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


class ListMessagesRequest(BaseModel):
    days_back: int = 7
    max_results: int = 20


class GetMessageRequest(BaseModel):
    message_id: str


class SendMessageRequest(BaseModel):
    to: str
    subject: str
    body: str


class SaveDraftRequest(BaseModel):
    to: str
    subject: str
    body: str
    reply_to_message_id: str | None = None


@app.post("/gmail/messages")
def list_messages(req: ListMessagesRequest | None = None) -> dict[str, Any]:
    """List emails from inbox, filtered by recency."""
    if req is None:
        req = ListMessagesRequest()

    cutoff = datetime.now(timezone.utc) - timedelta(days=req.days_back)
    results = []
    for email in _emails:
        email_date = datetime.fromisoformat(email["date"].replace("Z", "+00:00"))
        if email_date >= cutoff:
            results.append({
                "message_id": email["message_id"],
                "from": email["from"],
                "subject": email["subject"],
                "date": email["date"],
                "is_read": email["is_read"],
                "labels": email["labels"],
            })
    results = results[: req.max_results]

    resp = {"messages": results, "total": len(results)}
    _log_call("/gmail/messages", req.model_dump(), resp)
    return resp


@app.post("/gmail/messages/get")
def get_message(req: GetMessageRequest) -> dict[str, Any]:
    """Get a single email by message_id."""
    for email in _emails:
        if email["message_id"] == req.message_id:
            resp = copy.deepcopy(email)
            _log_call("/gmail/messages/get", req.model_dump(), resp)
            return resp

    resp = {"error": f"Message {req.message_id} not found"}
    _log_call("/gmail/messages/get", req.model_dump(), resp)
    return resp


@app.post("/gmail/send")
def send_message(req: SendMessageRequest) -> dict[str, Any]:
    """Send an email. Reactive replies are triggered when sending to configured recipients."""
    msg = {
        "to": req.to,
        "subject": req.subject,
        "body": req.body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _sent_messages.append(msg)

    to_lower = req.to.strip().lower()
    replies_list: list[dict[str, Any]] | None = None
    for key, val in _reactive_replies.items():
        if key.strip().lower() == to_lower:
            replies_list = val
            break

    if replies_list is not None:
        count = _send_counts.get(to_lower, 0)
        _send_counts[to_lower] = count + 1
        if count < len(replies_list):
            reply = copy.deepcopy(replies_list[count])
            reply["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            reply.setdefault("is_read", False)
            reply.setdefault("labels", ["inbox"])
            _emails.append(reply)
            _triggered_replies.append({
                "trigger_to": req.to,
                "send_count": count + 1,
                "reply_message_id": reply.get("message_id", ""),
                "timestamp": reply["date"],
            })
    else:
        _send_counts[to_lower] = _send_counts.get(to_lower, 0) + 1

    resp = {"status": "sent", "message": msg}
    _log_call("/gmail/send", req.model_dump(), resp)
    return resp


@app.post("/gmail/drafts/save")
def save_draft(req: SaveDraftRequest) -> dict[str, Any]:
    """Save an email as draft (not sent)."""
    draft = {
        "to": req.to,
        "subject": req.subject,
        "body": req.body,
        "reply_to_message_id": req.reply_to_message_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _drafts.append(draft)
    resp = {"status": "draft_saved", "draft": draft}
    _log_call("/gmail/drafts/save", req.model_dump(), resp)
    return resp


@app.get("/gmail/audit")
def get_audit() -> dict[str, Any]:
    """Return all API calls for grader inspection."""
    return {
        "calls": _audit_log,
        "sent_messages": _sent_messages,
        "drafts": _drafts,
        "triggered_replies": _triggered_replies,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9100")))
