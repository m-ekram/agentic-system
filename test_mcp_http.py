"""
test_mcp_http.py — the remote transport's front door.

The tools and the approval gate behind this transport are already covered by
test_mcp_server.py. What's new on the HTTP path is what stands in front of
them, so that's what these pin down: no token, no entry; the wrong host, no
entry; and the right token actually reaches a live, stateful MCP session —
the kind elicitation needs.
"""

import pytest
from fastapi.testclient import TestClient
from mcp_types.version import LATEST_PROTOCOL_VERSION

from mcp_server import MIN_TOKEN_LENGTH, build_http_app

TOKEN = "t" * MIN_TOKEN_LENGTH
HOST = "testserver"  # the Host header TestClient sends by default

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {"elicitation": {}},
        "clientInfo": {"name": "test", "version": "0"},
    },
}

# Streamable HTTP clients must accept both; the server rejects the POST otherwise.
ACCEPT = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def client():
    # Used as a context manager so the lifespan runs. That is what starts the
    # MCP session manager, and the token middleware must pass it through.
    with TestClient(build_http_app(TOKEN, HOST)) as c:
        yield c


def _post(client, **headers):
    return client.post("/mcp", json=INITIALIZE, headers={**ACCEPT, **headers})


def test_missing_token_is_rejected(client):
    assert _post(client).status_code == 401


def test_wrong_token_is_rejected(client):
    assert _post(client, Authorization="Bearer " + "x" * MIN_TOKEN_LENGTH).status_code == 401


def test_token_without_the_bearer_scheme_is_rejected(client):
    assert _post(client, Authorization=TOKEN).status_code == 401


def test_right_token_reaches_a_stateful_session(client):
    response = _post(client, Authorization=f"Bearer {TOKEN}")

    assert response.status_code == 200
    # A session id means stateful mode: the channel elicitation travels on.
    # Stateless mode sends none, and every write would be denied.
    assert response.headers.get("mcp-session-id")


def test_request_addressed_to_another_host_is_rejected(client):
    """DNS rebinding protection has to be on even though we bind 0.0.0.0."""
    response = _post(client, Authorization=f"Bearer {TOKEN}", Host="evil.example")
    assert response.status_code == 421


@pytest.mark.parametrize("token", ["", "short"])
def test_refuses_to_start_with_a_weak_token(token):
    with pytest.raises(ValueError, match="MCP_TOKEN"):
        build_http_app(token, HOST)


def test_refuses_to_start_without_a_public_host():
    with pytest.raises(ValueError, match="MCP_PUBLIC_HOST"):
        build_http_app(TOKEN, "")
