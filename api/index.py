"""Vercel serverless entry — wraps the FastMCP Streamable HTTP app.

FastMCP's streamable HTTP transport needs its session-manager task group
running (normally started by the ASGI lifespan) before it can serve requests.
Serverless platforms don't reliably drive the lifespan protocol, so this
wrapper drives it manually on the first request; when the platform does send
lifespan events, they pass straight through.

Requires VERCEL/STATELESS_HTTP so the MCP layer runs stateless — each request
may hit a fresh instance, so no in-memory session can be assumed.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import build_app  # noqa: E402

_inner = build_app()
_started: asyncio.Event | None = None


async def _drive_lifespan() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "lifespan.startup"})

    async def receive():
        # First call returns startup; later calls block forever — the shutdown
        # message is never sent, which keeps the task group alive for the
        # lifetime of the instance.
        return await queue.get()

    async def send(message):
        if message["type"] in ("lifespan.startup.complete", "lifespan.startup.failed"):
            _started.set()

    await _inner({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)


async def app(scope, receive, send):
    global _started
    if scope["type"] == "lifespan":
        # Platform drives lifespan itself — pass through and trust it to wait
        # for startup.complete before routing requests.
        _started = asyncio.Event()
        _started.set()
        return await _inner(scope, receive, send)
    if _started is None:
        _started = asyncio.Event()
        asyncio.get_event_loop().create_task(_drive_lifespan())
    await _started.wait()
    await _inner(scope, receive, send)
