"""
test_mcp_server.py — guards against the MCP wrappers drifting from the registry.

mcp_server.py writes its six wrapper signatures out by hand, which is the one
place in the project where something is stated twice. These tests make the
duplication safe: add a tool to the registry and forget to expose it, or let an
argument name drift, and CI says so.
"""

from types import SimpleNamespace

import pytest

import audit
import mcp_server
import tools
from mcp_server import server


async def test_mcp_exposes_every_registered_tool():
    exposed = {t.name for t in await server.list_tools()}
    expected = {spec.name for spec in tools.TOOLS}
    assert exposed == expected, "mcp_server.py and the TOOLS registry disagree"


async def test_mcp_argument_names_match_the_pydantic_models():
    """A wrapper whose parameter names drift from the args model would fail at runtime."""
    by_name = {t.name: t for t in await server.list_tools()}

    for spec in tools.TOOLS:
        mcp_args = set(by_name[spec.name].input_schema.get("properties", {}))
        model_args = set(spec.args_model.model_fields)
        assert mcp_args == model_args, f"{spec.name} arguments drifted"


async def test_mcp_calls_reach_the_real_tools(notes_dir):
    result = await server.call_tool("read_note", {"filenames": ["todo.md"]})
    assert "buy oat milk" in result.content[0].text


async def test_mcp_write_is_still_confined_to_the_notes_folder(
    notes_dir, audit_log, monkeypatch
):
    """The allowlist lives in tools.py, so it protects this entry point too.

    Approval is granted here on purpose: this asserts the path check holds on
    its own, not that the gate happened to refuse first.
    """

    async def approve(ctx, tool_name, args):
        return True, "test"

    monkeypatch.setattr(mcp_server, "_ask_permission", approve)
    escaped = notes_dir.parent / "escaped.md"

    result = await server.call_tool(
        "write_note", {"filename": "../escaped.md", "content": "pwned"}
    )

    assert not escaped.exists()
    # And the caller is told why, rather than getting a bare "tool failed".
    assert "blocked" in result.content[0].text


async def test_mcp_fails_closed_with_no_session(notes_dir, audit_log):
    """A client that cannot be asked gets refused, not obeyed.

    call_tool outside a real MCP session has no way to elicit, which is the
    same position any client without elicitation support is in.
    """
    result = await server.call_tool("delete_note", {"filename": "todo.md"})

    assert (notes_dir / "todo.md").exists()
    assert "declined" in result.content[0].text


# ---------------------------------------------------------------------------
# The approval gate on the MCP path
# ---------------------------------------------------------------------------
#
# The decorated tools can't be called with a fake context — MCP injects the real
# one — so these drive _guarded() directly, which is where the gate actually
# lives. test_gate_is_actually_wired_up below closes the loop by proving the
# decorated tools route through it.


class FakeCtx:
    """Stands in for an MCP Context, scripting what the human answers.

    action="accept" with approve=False is a real case, not a contrived one: the
    client returned a response, and the answer inside it was no.
    """

    def __init__(self, action="accept", approve=True, unsupported=False):
        self.action = action
        self.approve = approve
        self.unsupported = unsupported
        self.messages: list[str] = []

    async def elicit(self, message, schema):
        self.messages.append(message)
        if self.unsupported:
            raise RuntimeError("client does not support elicitation")
        data = schema(approve=self.approve) if self.action == "accept" else None
        return SimpleNamespace(action=self.action, data=data)


def _delete(ctx, filename="todo.md"):
    return mcp_server._guarded(
        ctx,
        "delete_note",
        {"filename": filename},
        lambda: tools.delete_note(tools.DeleteNoteArgs(filename=filename)),
    )


def _write(ctx, filename="new.md", content="hello"):
    return mcp_server._guarded(
        ctx,
        "write_note",
        {"filename": filename, "content": content},
        lambda: tools.write_note(
            tools.WriteNoteArgs(filename=filename, content=content)
        ),
    )


async def test_mcp_delete_asks_before_acting(notes_dir, audit_log):
    ctx = FakeCtx(approve=True)

    result = await _delete(ctx)

    assert not (notes_dir / "todo.md").exists()
    assert "delete_note" in ctx.messages[0], "the human was not shown the action"
    assert "todo.md" in ctx.messages[0], "the human was not shown the target"
    assert "Deleted" in result


async def test_mcp_delete_is_blocked_when_declined(notes_dir, audit_log):
    """The regression this whole gate exists to prevent."""
    result = await _delete(FakeCtx(approve=False))

    assert (notes_dir / "todo.md").exists(), "note deleted despite refusal"
    assert "declined" in result


@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_mcp_treats_decline_and_cancel_as_no(notes_dir, audit_log, action):
    await _delete(FakeCtx(action=action))
    assert (notes_dir / "todo.md").exists()


async def test_mcp_fails_closed_without_elicitation_support(
    notes_dir, audit_log, monkeypatch
):
    """If we can't reach a human, we don't change anything."""
    monkeypatch.setattr(mcp_server, "FALLBACK", "deny")

    result = await _delete(FakeCtx(unsupported=True))

    assert (notes_dir / "todo.md").exists()
    assert "declined" in result
    assert audit.read_events()[0]["gate"] == "denied-no-elicitation-support"


async def test_mcp_client_fallback_is_recorded_as_weaker(
    notes_dir, audit_log, monkeypatch
):
    """Opting into the client's own gate is allowed, but the log says so."""
    monkeypatch.setattr(mcp_server, "FALLBACK", "client")

    await _delete(FakeCtx(unsupported=True))

    assert not (notes_dir / "todo.md").exists()
    assert audit.read_events()[0]["gate"] == "delegated-to-client"


async def test_mcp_logs_approval_and_state_change(notes_dir, audit_log):
    await _write(FakeCtx(approve=True))

    events = audit.read_events()
    kinds = [(e["event_type"], e.get("decision") or e.get("action")) for e in events]
    assert ("approval", "approved") in kinds
    assert ("state_change", "write_note") in kinds
    # The record says which entry point acted, so the CLI and MCP paths stay
    # distinguishable after the fact.
    assert events[0]["via"] == "mcp"


async def test_mcp_declined_write_logs_no_state_change(notes_dir, audit_log):
    await _write(FakeCtx(approve=False))

    events = audit.read_events()
    assert [e["event_type"] for e in events] == ["approval"]
    assert not (notes_dir / "new.md").exists()


async def test_mcp_read_only_tools_need_no_approval(notes_dir, audit_log):
    """Reads must not interrupt the user, and must not appear in the approval log."""
    assert "todo.md" in await mcp_server.list_notes()
    assert "oat milk" in await mcp_server.read_note(["todo.md"])
    assert "todo.md" in await mcp_server.search_notes("oat")
    assert audit.read_events() == []


async def test_mcp_approved_write_still_cannot_escape_notes_folder(
    notes_dir, audit_log
):
    """Approval and the allowlist are independent: saying yes doesn't unlock the path."""
    escaped = notes_dir.parent / "escaped.md"

    result = await _write(FakeCtx(approve=True), filename="../escaped.md", content="pwned")

    assert not escaped.exists()
    assert "blocked" in result


async def test_gate_is_actually_wired_up(notes_dir, audit_log, monkeypatch):
    """The decorated tools must route through the gate, not just define one.

    Without this, every test above could pass while write_note and delete_note
    quietly bypassed _guarded entirely — which is exactly the bug this whole
    change exists to fix.
    """
    asked = []

    async def refuse(ctx, tool_name, args):
        asked.append(tool_name)
        return False, "test"

    monkeypatch.setattr(mcp_server, "_ask_permission", refuse)

    await server.call_tool("delete_note", {"filename": "todo.md"})
    await server.call_tool("write_note", {"filename": "x.md", "content": "y"})

    assert asked == ["delete_note", "write_note"], "a risky tool skipped the gate"
    assert (notes_dir / "todo.md").exists()
    assert not (notes_dir / "x.md").exists()


async def test_read_only_tools_do_not_ask(notes_dir, audit_log, monkeypatch):
    """Mirror of the above: reads must never reach the gate."""
    asked = []

    async def refuse(ctx, tool_name, args):
        asked.append(tool_name)
        return False, "test"

    monkeypatch.setattr(mcp_server, "_ask_permission", refuse)

    await server.call_tool("list_notes", {})
    await server.call_tool("read_note", {"filenames": ["todo.md"]})

    assert asked == []
