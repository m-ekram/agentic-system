"""
test_mcp_server.py — guards against the MCP wrappers drifting from the registry.

mcp_server.py writes its six wrapper signatures out by hand, which is the one
place in the project where something is stated twice. These tests make the
duplication safe: add a tool to the registry and forget to expose it, or let an
argument name drift, and CI says so.
"""

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


async def test_mcp_write_is_still_confined_to_the_notes_folder(notes_dir, audit_log):
    """The allowlist lives in tools.py, so it protects this entry point too."""
    escaped = notes_dir.parent / "escaped.md"

    result = await server.call_tool(
        "write_note", {"filename": "../escaped.md", "content": "pwned"}
    )

    assert not escaped.exists()
    # And the caller is told why, rather than getting a bare "tool failed".
    assert "blocked" in result.content[0].text
