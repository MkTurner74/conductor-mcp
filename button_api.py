"""
Plain REST routes for the Cantemo Portal "Create LoRA" button.

These exist because a Django view calling MCP JSON-RPC for a fixed,
button-click flow is unnecessary complexity -- the plugin backend is not an
agent deciding what to call, it always does the same three things. See
projects/coreweave-ibc-lora-demo/README.md for the full design.

No new pipeline logic: this is a thin orchestration layer over functions
that already exist and are already proven (lora_pipeline.py, used by the
MCP tools in server.py). Auth is the same shared bearer token as the MCP
surface -- these routes sit behind the same BearerAuthMiddleware.
"""

import json
import os

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import lora_pipeline

_DEFAULT_MODEL = "sdxl"


async def _read_json(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}


async def lora_train(request: Request) -> JSONResponse:
    """
    POST /lora/train
    Body: {item_ids: [str], label: str, trigger_word?, class_word?, model?, epochs?}

    Submits the real training job (not a dry run -- this endpoint exists to
    be called from a live "Create LoRA" click) and creates the MAM item
    immediately so the round trip is visible in Cantemo while the GPU job
    runs. Returns fast; does not wait for training to finish.
    """
    body = await _read_json(request)
    item_ids = body.get("item_ids") or []
    label = body.get("label")
    if not item_ids or not label:
        return JSONResponse({"error": "item_ids and label are required"}, status_code=400)

    trigger_word = body.get("trigger_word", "sks")
    class_word = body.get("class_word", "style")
    model = body.get("model", _DEFAULT_MODEL)
    epochs = int(body.get("epochs", 10))

    workdir = os.path.join(
        lora_pipeline.default_workdir(),
        "".join(ch if ch.isalnum() else "-" for ch in label).strip("-").lower(),
    )

    try:
        submit_result = await lora_pipeline.submit_training(
            dry_run=False,
            item_ids=item_ids,
            label=label,
            workdir=workdir,
            model=model,
            trigger_word=trigger_word,
            class_word=class_word,
            epochs=epochs,
        )
        job_id = submit_result.get("jid")
        if not job_id:
            return JSONResponse(
                {"error": "Conductor submission returned no job id", "detail": submit_result},
                status_code=502,
            )

        tracked = await lora_pipeline.create_tracked_lora_item(
            label=label,
            job_id=job_id,
            source_item_ids=item_ids,
            base_model=f"{model}1-kohya",
            trigger_word=trigger_word,
            created_by="Cantemo Portal button",
        )
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return JSONResponse(
        {
            "job_id": job_id,
            "item_id": tracked.get("item_id"),
            "tracked": tracked,
            "submit": submit_result,
        }
    )


async def lora_status(request: Request) -> JSONResponse:
    """
    GET /lora/status/{item_id}?job_id=...

    One-shot status check, meant to be polled client-side (the plugin's JS,
    same pattern as Samsyn's RunPanel -- no webhooks exist to push this).
    Syncs Conductor's job state onto the MAM item, and on the transition to
    "ready" also attaches the trained weights (finalize) so a single polling
    loop is enough to drive the whole round trip to completion.
    """
    item_id = request.path_params["item_id"]
    job_id = request.query_params.get("job_id")
    if not job_id:
        return JSONResponse({"error": "job_id query param is required"}, status_code=400)

    try:
        synced = await lora_pipeline.sync_status_to_mam(item_id, job_id, kind="lora")
        result = {"item_id": item_id, "job_id": job_id, **synced}

        if synced.get("mam_status") == lora_pipeline.STATUS_READY:
            finalized = await lora_pipeline.finalize_tracked_lora(item_id, job_id)
            result["finalized"] = finalized
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return JSONResponse(result)


routes = [
    Route("/lora/train", lora_train, methods=["POST"]),
    Route("/lora/status/{item_id}", lora_status, methods=["GET"]),
]
