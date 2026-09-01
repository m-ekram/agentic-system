"""
test_api.py — tests for the read-only audit log viewer.

Small surface, but two things here are worth pinning: the filters actually
filter, and the API stays read-only. The audit log's value depends on nothing
being able to rewrite it, so a route that could is a bug even if it works.
"""

from fastapi.testclient import TestClient

import audit
from api import app

client = TestClient(app)


def _seed():
    """One of each event type, so the filters have something to tell apart."""
    audit.append_event("approval", action="write_note", args={"filename": "a.md"}, decision="approved")
    audit.append_event("state_change", action="write_note", filename="a.md")
    audit.append_event("approval", action="delete_note", args={"filename": "b.md"}, decision="declined")
    audit.append_event("escalation", reason="max iterations", iterations=9)


def test_audit_log_returns_every_event(audit_log):
    _seed()

    body = client.get("/audit-log").json()

    assert body["total"] == 4
    assert [e["event_type"] for e in body["events"]] == [
        "approval",
        "state_change",
        "approval",
        "escalation",
    ]


def test_audit_log_is_empty_before_anything_happens(audit_log):
    body = client.get("/audit-log").json()
    assert body == {"total": 0, "events": []}


def test_limit_returns_the_most_recent_events(audit_log):
    _seed()

    body = client.get("/audit-log", params={"limit": 2}).json()

    # total still reports the real count, so a truncated view can't be mistaken
    # for the whole log.
    assert body["total"] == 4
    assert len(body["events"]) == 2
    assert body["events"][-1]["event_type"] == "escalation"


def test_state_changes_filter(audit_log):
    _seed()

    body = client.get("/audit-log/state-changes").json()

    assert body["total"] == 1
    assert body["events"][0]["action"] == "write_note"


def test_declined_filter_shows_only_refusals(audit_log):
    """The refusals are the interesting half of a human-in-the-loop log."""
    _seed()

    body = client.get("/audit-log/declined").json()

    assert body["total"] == 1
    assert body["events"][0]["args"] == {"filename": "b.md"}


def test_api_is_read_only(audit_log):
    """No route may create, modify or delete anything.

    An append-only log guarded by an API that can rewrite it is not append-only.
    """
    unsafe = {"POST", "PUT", "PATCH", "DELETE"}
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        assert not (methods & unsafe), f"{route.path} exposes {methods & unsafe}"
