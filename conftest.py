"""
conftest.py — shared test fixtures, including the fake LLM client.

The important idea in this file is the fake client. Every eval runs against a
*scripted* sequence of model responses instead of a real API. That means:

  - tests are deterministic; a model having an off day can't turn CI red
  - tests are instant and free, so CI needs no API key and no secret
  - we can assert on the exact tool sequence the loop produced

What that buys us is a clear division of labour: these tests check *our control
flow* — does the loop stop, does the gate hold, does the log get written — not
whether the model is smart. Model quality is a different question and it isn't
what a regression test should be measuring.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import audit
import tools


# ---------------------------------------------------------------------------
# Minimal stand-ins for the OpenAI response objects
# ---------------------------------------------------------------------------


@dataclass
class FakeFunction:
    name: str
    # A JSON *string*, exactly as the real API sends it.
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeResponse:
    choices: list = field(default_factory=list)


def text_turn(text: str) -> FakeResponse:
    """A turn where the model answers in words and stops."""
    return FakeResponse(choices=[SimpleNamespace(message=FakeMessage(content=text))])


def tool_turn(*calls: tuple[str, dict]) -> FakeResponse:
    """A turn where the model asks for one or more tools.

    tool_turn(("read_note", {"filenames": ["a.md"]}))
    """
    import json

    tool_calls = [
        FakeToolCall(id=f"call_{i}", function=FakeFunction(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    return FakeResponse(
        choices=[SimpleNamespace(message=FakeMessage(content=None, tool_calls=tool_calls))]
    )


class FakeClient:
    """Replays a scripted list of responses and records what it was asked."""

    def __init__(self, script: list[FakeResponse], repeat_last: bool = False):
        self.script = list(script)
        self.repeat_last = repeat_last
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, tools):
        # Snapshot the conversation so tests can assert on protocol correctness
        # (e.g. that every tool_call got exactly one matching tool reply).
        self.requests.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools})

        if not self.script:
            raise AssertionError("fake client ran out of scripted responses")
        if self.repeat_last and len(self.script) == 1:
            return self.script[0]
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    """Point the tools at a throwaway notes folder with two notes in it.

    Every test gets a fresh directory, so nothing a test writes or deletes can
    leak into the real notes/ folder or into another test.
    """
    d = tmp_path / "notes"
    d.mkdir()
    (d / "welcome.md").write_text("# Welcome\nThis is the welcome note.\n", encoding="utf-8")
    (d / "todo.md").write_text("# Todo\n- buy oat milk\n- write tests\n", encoding="utf-8")
    monkeypatch.setattr(tools, "NOTES_DIR", d.resolve())
    return d


@pytest.fixture
def audit_log(tmp_path, monkeypatch):
    """Redirect the audit log to a temp file and hand back a reader."""
    path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", path)
    return path


@pytest.fixture
def approvals():
    """A recording approver whose answer the test controls.

    approvals.answer = False  ->  decline everything
    approvals.seen            ->  list of (tool_name, args) it was asked about
    """

    class Approvals:
        def __init__(self):
            self.answer = True
            self.seen: list[tuple[str, dict]] = []

        async def __call__(self, tool_name: str, args: dict) -> bool:
            self.seen.append((tool_name, args))
            return self.answer

    return Approvals()
