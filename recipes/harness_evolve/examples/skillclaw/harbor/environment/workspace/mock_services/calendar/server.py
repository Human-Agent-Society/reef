"""Mock Calendar API service for meeting negotiation agent evaluation (FastAPI on port 9101)."""

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

app = FastAPI(title="Mock Calendar API")

from _base import add_error_injection
add_error_injection(app)

FIXTURES_PATH = Path(os.environ.get(
    "CALENDAR_FIXTURES",
    str(Path(__file__).resolve().parent.parent
        / "fixtures" / "calendar" / "events.json"),
))

_events: list[dict[str, Any]] = []
_audit_log: list[dict[str, Any]] = []
_deleted: list[dict[str, Any]] = []
_created_events: list[dict[str, Any]] = []


def _load_fixtures() -> None:
    global _events, _audit_log, _deleted, _created_events

    _audit_log = []
    _deleted = []
    _created_events = []

    with open(FIXTURES_PATH) as f:
        _events = json.load(f)


_load_fixtures()


def _log_call(endpoint: str, request_body: dict[str, Any], response_body: Any) -> None:
    _audit_log.append({
        "endpoint": endpoint,
        "request_body": request_body,
        "response_body": response_body,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


class ListEventsRequest(BaseModel):
    date: str
    days: int = 1


class GetEventRequest(BaseModel):
    event_id: str


class CreateEventRequest(BaseModel):
    title: str
    start_time: str
    end_time: str
    attendees: list[str] = Field(default_factory=list)
    location: str = ""


class DeleteEventRequest(BaseModel):
    event_id: str


@app.post("/calendar/events")
def list_events(req: ListEventsRequest | None = None) -> dict[str, Any]:
    """List events within a date range."""
    if req is None:
        req = ListEventsRequest(date="2026-03-02")
    try:
        query_date = datetime.strptime(req.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        resp = {"error": f"Invalid date format: {req.date}"}
        _log_call("/calendar/events", req.model_dump(), resp)
        return resp

    end_date = query_date + timedelta(days=req.days)
    results = []
    for evt in _events:
        evt_start = datetime.fromisoformat(evt["start_time"].replace("Z", "+00:00"))
        if query_date <= evt_start < end_date:
            results.append(copy.deepcopy(evt))
    results.sort(key=lambda e: e["start_time"])
    resp = {"events": results, "total": len(results)}
    _log_call("/calendar/events", req.model_dump(), resp)
    return resp


@app.post("/calendar/events/get")
def get_event(req: GetEventRequest) -> dict[str, Any]:
    """Get a single event by event_id."""
    for evt in _events:
        if evt["event_id"] == req.event_id:
            resp = copy.deepcopy(evt)
            _log_call("/calendar/events/get", req.model_dump(), resp)
            return resp
    resp = {"error": f"Event {req.event_id} not found"}
    _log_call("/calendar/events/get", req.model_dump(), resp)
    return resp


@app.post("/calendar/events/create")
def create_event(req: CreateEventRequest) -> dict[str, Any]:
    """Create a new calendar event."""
    new_id = f"evt_{len(_events) + 1:03d}"
    event = {
        "event_id": new_id,
        "title": req.title,
        "start_time": req.start_time,
        "end_time": req.end_time,
        "attendees": req.attendees,
        "location": req.location,
        "recurring": False,
    }
    _events.append(event)
    _created_events.append(copy.deepcopy(event))
    resp = {"status": "created", "event": event}
    _log_call("/calendar/events/create", req.model_dump(), resp)
    return resp


@app.post("/calendar/events/delete")
def delete_event(req: DeleteEventRequest) -> dict[str, Any]:
    """Delete an event by event_id."""
    for i, evt in enumerate(_events):
        if evt["event_id"] == req.event_id:
            removed = _events.pop(i)
            _deleted.append(removed)
            resp = {"status": "deleted", "event": removed}
            _log_call("/calendar/events/delete", req.model_dump(), resp)
            return resp
    resp = {"error": f"Event {req.event_id} not found"}
    _log_call("/calendar/events/delete", req.model_dump(), resp)
    return resp


@app.get("/calendar/audit")
def get_audit() -> dict[str, Any]:
    """Return all API calls for grader inspection."""
    return {"calls": _audit_log, "deleted": _deleted, "created_events": _created_events}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9101")))
