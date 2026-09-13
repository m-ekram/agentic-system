"""
audit.py — an append-only record of everything the agent did.

Why a plain .jsonl file and not a database:

An audit log needs two properties. It has to be *ordered* (what happened, in
what sequence) and it has to be *append-only* (you can add to it, you never
rewrite it — otherwise it's not evidence of anything). Opening a file in "a"
mode and writing one JSON object per line gives you both as a property of the
file itself, not as a promise made by some API on top of it.

A database would buy us concurrent writers, indexes and queries. This is a
single-user tool running on one laptop. We don't have those problems, and
picking SQLite here would have hidden the append-only guarantee behind an ORM
instead of making it the obvious thing it is.

Two kinds of event get written:
  - "approval"     — a human was asked about a risky action, and said y or n.
  - "state_change" — something on disk actually changed.

A declined action produces an approval event and NO state_change event. Read
together, the log tells you what was proposed, what was allowed, and what
actually happened.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module-level so the tests can point it at a temp file with monkeypatch, and
# overridable from the environment so a deployment can keep it on a volume.
AUDIT_LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", Path(__file__).parent / "audit_log.jsonl"))


def append_event(event_type: str, **fields: Any) -> dict:
    """Append exactly one JSON line to the audit log. Never rewrites."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **fields,
    }
    # "a" is the whole security model of this file: the OS opens at the end and
    # every write goes after what's already there. There is no code path in this
    # project that opens the log for writing any other way.
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_events() -> list[dict]:
    """Read the log back. Used by the tests and the FastAPI viewer."""
    if not Path(AUDIT_LOG_PATH).exists():
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
