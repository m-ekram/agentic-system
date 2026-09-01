"""
api.py — a read-only web view of the audit log.

Deliberately tiny, and deliberately read-only. The audit log's value comes from
being append-only, so the last thing it needs is an HTTP endpoint that can
modify it. There is no route here that writes anything.

Run it:  uvicorn api:app --reload
Then:    http://localhost:8000/audit-log
"""

from fastapi import FastAPI

import audit

app = FastAPI(
    title="Notes agent audit log",
    description="Read-only view of every action the agent proposed and performed.",
)


@app.get("/audit-log")
def get_audit_log(limit: int = 100) -> dict:
    """Return the most recent audit events, newest last."""
    events = audit.read_events()
    return {"total": len(events), "events": events[-limit:]}


@app.get("/audit-log/state-changes")
def get_state_changes() -> dict:
    """Just the actions that actually changed a file."""
    events = [e for e in audit.read_events() if e["event_type"] == "state_change"]
    return {"total": len(events), "events": events}


@app.get("/audit-log/declined")
def get_declined() -> dict:
    """Everything a human refused — the interesting half of a human-in-the-loop log."""
    events = [
        e
        for e in audit.read_events()
        if e["event_type"] == "approval" and e.get("decision") == "declined"
    ]
    return {"total": len(events), "events": events}
