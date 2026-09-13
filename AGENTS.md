# Agent guidance

This repo is a notes agent whose MCP server can run over authenticated HTTPS on EC2. The `deploy-ec2` branch (PR #1) adds that HTTP transport and the `deploy/` setup, and the live server runs that branch.

Before touching AWS or the server (SSH, deploys, logs, token rotation, any resource change), get the owner's handoff document. Resource IDs, the SSH command, the runbook, and the owner's open decisions live there, outside the repo.

## The approval gate

The gate is the point of the project: every write and delete waits for a human through MCP elicitation, and the server fails closed when it cannot ask. Keep these true in every change:

- Sessions stay stateful (`stateless_http=False`); elicitation travels on the session.
- The server keeps `NOTES_MCP_APPROVAL_FALLBACK` at its default, `deny`.
- Caddy is the only thing facing the internet; the app's port 8000 stays on the compose network.
- `api.py` stays off the internet, because approval events carry full note content.
- stdio stays the default transport, so the desktop setup keeps working.

The reasoning is in README.md, under "Whose gate is it?" and "As a remote MCP server (EC2)".

## Git and GitHub

- Commits and PRs read as the owner's own work: no co-author trailers, no generated-with footers, and no naming of any assistant or tool that helped write them.
- Match the history: an imperative subject, a body that explains why, and an `N tests (was M).` line when the test count changes.
- Stage explicit paths. Changes go through a branch and a PR, and both CI jobs (`test`, `docker`) are green before merge.

## Secrets

The bearer token lives in `deploy/.env` on the server and in the owner's MCP client config, nowhere else. Keep it out of files, commits, and anything you publish; grep a new document for it before sharing.

## Done means

- Code change: `pytest` passes, run from the project's virtualenv.
- Deployed change: rebuilt on the server, and `curl -i https://<MCP_PUBLIC_HOST>/mcp` returns 401.

## Gotchas

- SSH goes only through the EC2 Instance Connect Endpoint. The owner's connection sits behind carrier-grade NAT, so IP-based SSH rules fail.
- The AWS CLI signs in through the owner's browser session and expires with it; renewing it needs the owner.
- Piping a script to `ssh` from PowerShell adds a trailing CR: run it remotely as `tr -d '\r' | bash -s`. Inside a piped script, give `docker compose exec -T` `</dev/null`, or it consumes the rest of the script.
- AWS changes spend the owner's credits or change security: confirm each one with the owner first.
