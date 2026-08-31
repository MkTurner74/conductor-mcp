"""
Background job submission — so a caller never waits on the upload.

Submitting a LoRA training job is not a quick call. It pulls the training
images out of Cantemo, lays them out as a kohya dataset, then hands them to
ciocore's uploader (md5 + chunking, a worker pool). For eight stills that is
tens of seconds; for a real training set it is minutes. Anything calling this
over HTTP has to hold a request open for the whole of it.

Samsyn's run engine is serverless, so it cannot. A request there has a hard
ceiling measured in seconds, and a run must survive the process that started
it disappearing. So submission is split in two:

    submit_lora_training(background=True) -> {"submission_id": ...}    (instant)
    get_submission_status(submission_id)  -> {"state": ..., "jid": ...}

This container is long-lived, so the work runs here as an asyncio task while
the caller goes away and comes back.

**The restart hole, and how it is closed.** The registry below is in memory: a
Railway redeploy mid-upload loses it, and the caller would be left polling an
id nothing remembers. Rather than add a database for a few minutes of state,
`get_submission_status` falls back to asking CONDUCTOR whether a job by that
label exists. Conductor is the real system of record for "did this job get
submitted"; the registry is only a fast path. So a restart costs a slower
answer, never a wrong one, and never a duplicate GPU job.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

import conductor_client as conductor

_logger = logging.getLogger(__name__)

# submission_id -> record. Bounded by _prune(); a submission is interesting for
# as long as a run might still be polling it, not forever.
_registry: dict[str, dict[str, Any]] = {}
_RETAIN_SECONDS = 6 * 60 * 60


def _prune() -> None:
    cutoff = time.time() - _RETAIN_SECONDS
    for key in [k for k, v in _registry.items() if v.get("updated", 0) < cutoff]:
        _registry.pop(key, None)


def start(label: str, coro_factory: Callable[[], Awaitable[dict]]) -> str:
    """
    Run a submission in the background and return its ticket immediately.

    `coro_factory` is called here rather than being passed an already-created
    coroutine so that nothing begins executing until there is a registry entry
    to record it against.
    """
    _prune()
    submission_id = f"sub-{uuid.uuid4().hex[:12]}"
    _registry[submission_id] = {
        "state": "staging",
        "label": label,
        "jid": None,
        "error": None,
        "result": None,
        "created": time.time(),
        "updated": time.time(),
    }

    async def runner() -> None:
        rec = _registry.get(submission_id)
        try:
            result = await coro_factory()
        except Exception as exc:  # noqa: BLE001 — the ticket must record every failure
            _logger.exception("[submissions] %s failed", submission_id)
            if rec is not None:
                rec.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated=time.time())
            return
        if rec is None:
            return
        # A dry run is a legitimate terminal state with no job — do not let it
        # look like a submission that failed to produce one.
        if result.get("dry_run"):
            rec.update(state="dry_run", result=result, updated=time.time())
        elif result.get("error"):
            rec.update(state="failed", error=str(result["error"]), result=result, updated=time.time())
        elif result.get("jid"):
            rec.update(state="submitted", jid=str(result["jid"]), result=result, updated=time.time())
        else:
            rec.update(state="failed", error="submission returned no job id", result=result, updated=time.time())

    asyncio.create_task(runner())
    return submission_id


async def _find_job_by_label(label: str) -> Optional[str]:
    """
    Ask Conductor whether a job for this label exists — the restart fallback.

    Titles are built as "LoRA train — <label>" / "LoRA generate — <label>", so
    a substring match on the label is enough to recognise our own job. Newest
    wins: a re-run of the same label should resolve to the current attempt.
    """
    try:
        raw = await conductor.list_jobs()
    except Exception as exc:  # noqa: BLE001
        _logger.error("[submissions] label lookup failed: %s", exc)
        return None
    jobs = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(jobs, list):
        return None
    needle = label.strip().lower()
    matches = [j for j in jobs if needle and needle in str(j.get("title") or "").lower()]
    if not matches:
        return None
    matches.sort(key=lambda j: str(j.get("created") or ""))
    return str(matches[-1].get("jid") or "") or None


async def status(submission_id: str, label: str = "") -> dict:
    """
    Where a submission has got to.

    States: staging (uploading) · submitted (has a jid) · dry_run · failed ·
    unknown (this process has never heard of it and Conductor has no matching
    job — the caller should treat that as a failure, not keep waiting).
    """
    rec = _registry.get(submission_id)
    if rec is not None:
        return {
            "submission_id": submission_id,
            "state": rec["state"],
            "jid": rec.get("jid"),
            "error": rec.get("error"),
            "label": rec.get("label"),
            "result": rec.get("result"),
        }

    # Not in memory. Either the ticket is wrong, or this container restarted
    # after the job was already submitted — ask Conductor which.
    if label:
        jid = await _find_job_by_label(label)
        if jid:
            return {
                "submission_id": submission_id,
                "state": "submitted",
                "jid": jid,
                "recovered": True,
                "note": "Ticket lost (server restarted); job recovered from Conductor by label.",
            }
    return {
        "submission_id": submission_id,
        "state": "unknown",
        "error": "No such submission on this server, and no matching Conductor job.",
    }
