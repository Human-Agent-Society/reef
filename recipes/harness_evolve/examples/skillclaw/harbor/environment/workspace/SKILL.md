---
name: 03_task1
description: "Coordinates meeting scheduling via Gmail and Calendar APIs. Use when: scheduling a meeting, checking participant availability, sending coordination emails, creating calendar events, or notifying organizers of confirmed bookings."
---

# Scheduling Assistant Skill

Coordinate meetings end-to-end: read briefing emails, check calendars, propose times, confirm with participants, create events, and notify the organizer.

## Tools

All tools are defined in `tmp_workspace/utils.py`:

- **`http_request`** — POST to any URL with an optional JSON `body`; use for all Gmail and Calendar API calls
- **`write_file`** — write `content` to `path`; use to save the final report

---

## Gmail API

**Base URL:** `http://localhost:9100`

| Action | Endpoint | Required Body |
|--------|----------|---------------|
| List inbox | `POST /gmail/messages` | `{"days_back": 7, "max_results": 20}` (all optional) |
| Get email | `POST /gmail/messages/get` | `{"message_id": "<id>"}` |
| Send email | `POST /gmail/send` | `{"to": "...", "subject": "...", "body": "..."}` |

> After sending any email, re-check the inbox for replies before proceeding.

---

## Calendar API

**Base URL:** `http://localhost:9101`

| Action | Endpoint | Required Body |
|--------|----------|---------------|
| List events | `POST /calendar/events` | `{"date": "YYYY-MM-DD", "days": 1}` (`date` required) |
| Create event | `POST /calendar/events/create` | `{"title": "...", "start_time": "...", "end_time": "...", "attendees": [...]}` + optional `"location"` |

---

## Workflow

1. **Read briefing** — check inbox for the organizer's original request email
2. **Check calendars** — list each participant's events for the candidate date range
3. **Propose a time** — email participants with an available slot matching the required duration
4. **Collect replies** — re-check inbox after each send to gather confirmations
5. **Create event** — once confirmed, create the calendar event with all attendees
6. **Notify organizer** — send a confirmation email to the original requester
7. **Write report** — save summary to `/tmp_workspace/results/results.md`

---

## Constraints

- Do **not** delete or cancel any participant's existing calendar events
- Final report must be written to `/tmp_workspace/results/results.md`