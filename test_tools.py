"""
test_tools.py — unit tests for each tool, in isolation, with no LLM involved.

These are the cheap tests: they prove the individual pieces behave, so that
test_agent_evals.py can concentrate on the loop's behaviour instead.
"""

import httpx
import pytest

import audit
import tools
from tools import (
    DeleteNoteArgs,
    FetchUrlArgs,
    ListNotesArgs,
    ReadNoteArgs,
    SearchNotesArgs,
    ToolError,
    WriteNoteArgs,
    delete_note,
    fetch_url,
    list_notes,
    read_note,
    search_notes,
    write_note,
)

# Paths a confused (or manipulated) model might ask for. None of them may work.
ESCAPE_ATTEMPTS = [
    "../escaped.md",
    "../../escaped.md",
    "../../../Windows/system32/evil.md",
    "notes/../../escaped.md",
    "subdir/../../escaped.md",
]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


async def test_list_notes(notes_dir):
    out = await list_notes(ListNotesArgs())
    assert "todo.md" in out
    assert "welcome.md" in out


async def test_list_notes_empty_folder(notes_dir):
    for p in notes_dir.glob("*.md"):
        p.unlink()
    assert "empty" in (await list_notes(ListNotesArgs())).lower()


async def test_read_note_reads_several_at_once(notes_dir):
    out = await read_note(ReadNoteArgs(filenames=["welcome.md", "todo.md"]))
    assert "This is the welcome note." in out
    assert "buy oat milk" in out


async def test_read_note_one_missing_file_does_not_sink_the_others(notes_dir):
    """A bad filename in a concurrent batch must not lose the good results."""
    out = await read_note(ReadNoteArgs(filenames=["welcome.md", "nope.md"]))
    assert "This is the welcome note." in out
    assert "no such note" in out


async def test_read_note_rejects_empty_list(notes_dir):
    with pytest.raises(ToolError):
        await read_note(ReadNoteArgs(filenames=[]))


@pytest.mark.parametrize("bad", ESCAPE_ATTEMPTS)
async def test_read_note_cannot_escape_the_notes_folder(notes_dir, bad):
    out = await read_note(ReadNoteArgs(filenames=[bad]))
    assert "blocked" in out


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


async def test_search_finds_a_match(notes_dir):
    out = await search_notes(SearchNotesArgs(query="oat milk"))
    assert "todo.md" in out


async def test_search_is_case_insensitive(notes_dir):
    assert "todo.md" in await search_notes(SearchNotesArgs(query="OAT MILK"))


async def test_search_with_no_match(notes_dir):
    out = await search_notes(SearchNotesArgs(query="xyzzy"))
    assert "No notes matched" in out


async def test_search_rejects_empty_query(notes_dir):
    with pytest.raises(ToolError):
        await search_notes(SearchNotesArgs(query="   "))


# ---------------------------------------------------------------------------
# Writing and deleting
# ---------------------------------------------------------------------------


async def test_write_creates_a_note(notes_dir, audit_log):
    await write_note(WriteNoteArgs(filename="idea.md", content="# Idea\nship it"))
    assert (notes_dir / "idea.md").read_text(encoding="utf-8") == "# Idea\nship it"


async def test_write_overwrite_is_reported(notes_dir, audit_log):
    out = await write_note(WriteNoteArgs(filename="todo.md", content="replaced"))
    assert "Overwrote" in out


async def test_write_rejects_non_markdown(notes_dir, audit_log):
    with pytest.raises(ToolError):
        await write_note(WriteNoteArgs(filename="script.py", content="import os"))


async def test_delete_removes_a_note(notes_dir, audit_log):
    await delete_note(DeleteNoteArgs(filename="todo.md"))
    assert not (notes_dir / "todo.md").exists()


async def test_delete_missing_note_errors(notes_dir, audit_log):
    with pytest.raises(ToolError):
        await delete_note(DeleteNoteArgs(filename="nope.md"))


# ---------------------------------------------------------------------------
# The allowlist — the security boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ESCAPE_ATTEMPTS)
async def test_write_cannot_escape_the_notes_folder(notes_dir, audit_log, bad):
    with pytest.raises(ToolError, match="blocked"):
        await write_note(WriteNoteArgs(filename=bad, content="pwned"))

    # And prove it: nothing was created anywhere above the notes folder.
    assert not (notes_dir.parent / "escaped.md").exists()


@pytest.mark.parametrize("bad", ESCAPE_ATTEMPTS)
async def test_delete_cannot_escape_the_notes_folder(notes_dir, audit_log, bad):
    victim = notes_dir.parent / "escaped.md"
    victim.write_text("do not delete me", encoding="utf-8")

    with pytest.raises(ToolError, match="blocked"):
        await delete_note(DeleteNoteArgs(filename=bad))

    assert victim.exists(), "a file outside the notes folder was deleted"


async def test_absolute_path_is_blocked(notes_dir, audit_log, tmp_path):
    outside = tmp_path / "outside.md"
    with pytest.raises(ToolError):
        await write_note(WriteNoteArgs(filename=str(outside), content="pwned"))
    assert not outside.exists()


# ---------------------------------------------------------------------------
# The audit log
# ---------------------------------------------------------------------------


async def test_every_state_change_is_logged(notes_dir, audit_log):
    await write_note(WriteNoteArgs(filename="a.md", content="one"))
    await delete_note(DeleteNoteArgs(filename="a.md"))

    events = audit.read_events()
    assert [e["action"] for e in events] == ["write_note", "delete_note"]
    assert all(e["event_type"] == "state_change" for e in events)


async def test_blocked_write_logs_nothing(notes_dir, audit_log):
    """A rejected action must not appear as a state change — because it wasn't one."""
    with pytest.raises(ToolError):
        await write_note(WriteNoteArgs(filename="../escaped.md", content="pwned"))
    assert audit.read_events() == []


async def test_log_is_append_only(notes_dir, audit_log):
    """Later writes add lines; they never rewrite the ones already there."""
    await write_note(WriteNoteArgs(filename="a.md", content="one"))
    first = audit_log.read_text(encoding="utf-8")

    await write_note(WriteNoteArgs(filename="b.md", content="two"))
    second = audit_log.read_text(encoding="utf-8")

    assert second.startswith(first), "existing audit lines were modified"
    assert len(second.splitlines()) == 2


# ---------------------------------------------------------------------------
# fetch_url: the network allowlist
# ---------------------------------------------------------------------------
#
# No test here touches the real network: a MockTransport stands in for it and
# records every request that would have gone out, so "blocked" can be proven by
# the request never being sent, not just by the error message.

# Where an injected "fetch this" might try to send the agent.
INTERNAL_URLS = [
    "http://127.0.0.1/",
    "http://localhost:8000/",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",
    "http://224.0.0.1/",
]


def _ok(request):
    return httpx.Response(200, text="hello")


def _fake_web(monkeypatch, handler, dns):
    """Serve `handler` instead of the network and resolve names from `dns`.

    Returns the list of URLs that actually reached the transport.
    """
    sent: list[str] = []

    def record(request):
        sent.append(str(request.url))
        return handler(request)

    async def resolve(host):
        if host not in dns:
            raise OSError(f"unknown host {host}")
        return dns[host]

    monkeypatch.setattr(tools, "_FETCH_TRANSPORT", httpx.MockTransport(record))
    monkeypatch.setattr(tools, "_resolve_host", resolve)
    return sent


@pytest.mark.parametrize("url", INTERNAL_URLS)
async def test_fetch_url_refuses_internal_addresses(monkeypatch, url):
    sent = _fake_web(monkeypatch, _ok, dns={"localhost": ["127.0.0.1", "::1"]})

    with pytest.raises(ToolError, match="blocked"):
        await fetch_url(FetchUrlArgs(url=url))

    assert sent == [], "a request to an internal address was sent"


async def test_fetch_url_blocks_localhost_with_the_real_resolver(monkeypatch):
    """The default resolver, not a fake one, must also catch a name."""
    monkeypatch.setattr(tools, "_FETCH_TRANSPORT", httpx.MockTransport(_ok))

    with pytest.raises(ToolError, match="blocked"):
        await fetch_url(FetchUrlArgs(url="http://localhost:8000/"))


async def test_fetch_url_blocks_a_name_with_any_internal_address(monkeypatch):
    """One internal address is enough to refuse: we can't pick which one gets used."""
    sent = _fake_web(
        monkeypatch, _ok, dns={"mixed.example.com": ["93.184.216.34", "10.0.0.5"]}
    )

    with pytest.raises(ToolError, match="blocked"):
        await fetch_url(FetchUrlArgs(url="https://mixed.example.com/"))

    assert sent == []


async def test_fetch_url_allows_public_hosts(monkeypatch):
    sent = _fake_web(monkeypatch, _ok, dns={"example.com": ["93.184.216.34"]})

    out = await fetch_url(FetchUrlArgs(url="https://example.com/"))

    assert "UNTRUSTED EXTERNAL CONTENT" in out
    assert "hello" in out
    assert sent == ["https://example.com/"]


async def test_fetch_url_blocks_a_redirect_to_an_internal_address(monkeypatch):
    """A public page must not be able to bounce the agent into the metadata service."""

    def handler(request):
        if request.url.host == "tips.example.com":
            return httpx.Response(
                302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, text="cloud credentials")

    sent = _fake_web(monkeypatch, handler, dns={"tips.example.com": ["93.184.216.34"]})

    with pytest.raises(ToolError, match="blocked"):
        await fetch_url(FetchUrlArgs(url="https://tips.example.com/"))

    assert sent == ["https://tips.example.com/"], "the redirected request was sent"


async def test_fetch_url_reports_an_unresolvable_host(monkeypatch):
    _fake_web(monkeypatch, _ok, dns={})

    with pytest.raises(ToolError, match="could not resolve"):
        await fetch_url(FetchUrlArgs(url="https://no-such-host.example/"))


async def test_fetch_url_rejects_non_http_schemes():
    with pytest.raises(ToolError, match="http"):
        await fetch_url(FetchUrlArgs(url="file:///etc/passwd"))
