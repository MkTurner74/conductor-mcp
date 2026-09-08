"""
Conductor for AI -> Cantemo MAM, the inference half.

The parallel path to lora_pipeline.py. Same destination (generated images
landing in the MAM with provenance), different compute door:

    lora_pipeline.py   kohya bash script inside a render job. Cold-starts the
                       model every time, stages the .safetensors up from the
                       MAM on every generation, minutes before a pixel exists.
    this module        POST /inferences against a resident model. No upload,
                       no package resolution, no bash.

Deliberately separate files. The IBC path is proven and locked; if this one
turns out to be wrong about the API, deleting it costs nothing.

Two things are NOT assumed here, because the OpenAPI document does not say:

  * the status vocabulary — now confirmed for the happy path (created, pending,
    running, completed) but still a superset below, because anything
    unrecognised is treated as "still running" rather than silently passing as
    success. A job that lies about finishing is the expensive failure (see the
    Conductor green-job lesson); a job that takes one extra poll is free.
  * whether a MAM-held .safetensors can be used here at all. This API generates
    with LoRAs IT holds (lora_details[].id is one of its own lora-models). A
    LoRA trained through the old kohya path lives in Cantemo as a file and is
    NOT addressable here until it is either retrained through this API or
    registered as an asset. That is the one real gap between the two paths, and
    it is why the first Samsyn test workflow generates from a base model with
    an OPTIONAL LoRA rather than requiring one.
"""

import logging
import os
from typing import Any, Optional

import cantemo_client as cantemo
import conductor_ai_client as ai
import lora_pipeline

_logger = logging.getLogger(__name__)

# Status spellings. CONFIRMED against inference 00010 (2026-09-08), which ran
# created -> pending -> running -> completed in 34 seconds. The rest are kept
# because this is an Open Beta and the vocabulary can grow; anything NOT listed
# reads as still-running, never as success.
TERMINAL_OK = {"success", "succeeded", "completed", "complete", "finished", "done"}
TERMINAL_BAD = {"failed", "failure", "error", "errored", "canceled", "cancelled",
                "killed", "terminated", "aborted", "timeout", "timed_out"}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Sensible generation defaults, matching the shape of the params the kohya path
# already exposes so a person moving between the two nodes is not relearning.
# Read off the live base models 2026-09-08, NOT carried over from the kohya
# path — and one of them is a trap. `prompt_adherence` here is a 1-4 scale
# (default 3), not the 7.5-and-up classifier-free-guidance number
# submit_lora_inference takes. Reusing 7.5 out of habit sends every generation
# a value past the model's own maximum.
DEFAULT_STEPS = 30          # models allow 5-50, default 50
DEFAULT_WIDTH = 1024        # 384/400 min, 1792/1808 max depending on model
DEFAULT_HEIGHT = 1024
DEFAULT_ADHERENCE = 3.0     # 1-4, default 3
DEFAULT_SEED = 42


async def project_id() -> str:
    """Resolved per call rather than cached at import: the key and project can
    be set after this process starts (they are, on Railway)."""
    return await ai.resolve_project_id()


# --- discovery -------------------------------------------------------------

def _model_row(row: dict) -> dict:
    return {
        "id": str(row.get("id") or ""),
        "name": str(row.get("name") or row.get("model_name") or ""),
        "model_name": str(row.get("model_name") or ""),
        "model_family": str(row.get("model_family") or ""),
        "published": bool(row.get("publish", False)),
    }


async def list_models() -> dict:
    pid = await project_id()
    resp = await ai.list_base_models(pid) or {}
    rows = resp.get("data") or []
    return {"project_id": pid, "models": [_model_row(r) for r in rows if isinstance(r, dict)]}


# The base models are named by their upstream release, not by the short name
# anyone actually says: SDXL is registered as "stable-diffusion-xl-base-1.0-…",
# which does NOT contain the string "sdxl". Without these aliases a node
# configured with the obvious word silently falls through to the first model in
# the list — a different model, generating perfectly good wrong pictures.
MODEL_ALIASES = {
    "sdxl": "stable-diffusion-xl",
    "sdxl1": "stable-diffusion-xl",
    "sd35-large": "stable-diffusion-3.5-large",
    "sd35l": "stable-diffusion-3.5-large",
    "sd35-medium": "stable-diffusion-3.5-medium",
    "sd35m": "stable-diffusion-3.5-medium",
    "sd3-medium": "stable-diffusion-3-medium",
    "sd3m": "stable-diffusion-3-medium",
    "flux": "FLUX.1-schnell",
    "flux-schnell": "FLUX.1-schnell",
    "fluxschnell": "FLUX.1-schnell",
}


async def resolve_model_id(model: str = "") -> str:
    """Accept an id, a name, or a family word like "sdxl" — and say what the
    options were when none of them match, rather than failing with a bare id."""
    listing = await list_models()
    models = listing["models"]
    if not models:
        raise ai.ConductorAIError(
            "GET", "/base-models", 404,
            "This Conductor-for-AI project has no base models registered. Register one with "
            "POST base-models, or ask Conductor which are pre-published for the account.")

    wanted = (model or os.getenv("CONDUCTOR_AI_BASE_MODEL", "") or "").strip()
    if not wanted:
        return models[0]["id"]

    low = MODEL_ALIASES.get(wanted.lower(), wanted).lower()
    for m in models:                                   # exact id first
        if m["id"] == wanted:
            return m["id"]
    for m in models:                                   # then exact name
        if low in (m["name"].lower(), m["model_name"].lower(), m["model_family"].lower()):
            return m["id"]
    for m in models:                                   # then a substring
        if low in f'{m["name"]} {m["model_name"]} {m["model_family"]}'.lower():
            return m["id"]
    raise ai.ConductorAIError(
        "GET", "/base-models", 404,
        f'No base model matching "{wanted}". Available: '
        + ", ".join(f'{m["name"] or m["model_name"]} ({m["id"]})' for m in models))


def _lora_row(row: dict) -> dict:
    return {
        "id": str(row.get("id") or ""),
        "name": str(row.get("name") or row.get("file_name") or ""),
        "trigger_word": str(row.get("trigger_word") or ""),
        "model_id": str(row.get("model_id") or ""),
        "model_name": str(row.get("model_name") or ""),
        "job_name": str(row.get("job_name") or ""),
        "epoch": row.get("epoch"),
        "total_epochs": row.get("total_epochs"),
    }


async def list_available_loras(model: str = "") -> dict:
    """LoRAs this API can generate with, for the given base model.

    Not the same population as cantemo_list_loras: that lists what the MAM
    holds, this lists what the inference endpoint can address. They only
    coincide once a LoRA has been trained through this API.
    """
    pid = await project_id()
    model_id = await resolve_model_id(model)
    resp = await ai.list_loras(pid, model_id) or {}
    rows = resp.get("data") or []
    return {
        "project_id": pid,
        "model_id": model_id,
        "loras": [_lora_row(r) for r in rows if isinstance(r, dict)],
    }


# --- generation ------------------------------------------------------------

def _machine(gpu_type: str = "", gpu_count: int = 0, memory: str = "") -> Optional[dict]:
    """settings.machine, only when the caller actually chose something.

    Sending a half-filled machine block is worse than sending none: the service
    picks a sane default, and a wrong GPU name is a submission-time rejection.
    """
    machine: dict = {}
    if gpu_type or gpu_count:
        gpu: dict = {}
        if gpu_type:
            gpu["type"] = gpu_type
        if gpu_count:
            gpu["count"] = int(gpu_count)
        machine["gpu"] = gpu
    if memory:
        machine["memory"] = memory
    return machine or None


async def submit_generation(
    prompt: str,
    model: str = "",
    lora_id: str = "",
    trigger_word: str = "",
    weight: float = 1.0,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    prompt_adherence: float = DEFAULT_ADHERENCE,
    name: str = "",
    gpu_type: str = "",
    gpu_count: int = 0,
) -> dict:
    """Start one generation and return immediately with its id.

    Unlike the kohya path this does NOT stage anything first, so there is no
    submission ticket to poll — the POST either creates a job or fails now.
    """
    if not prompt.strip():
        return {"ok": False, "error": "A prompt is required."}

    pid = await project_id()
    model_id = await resolve_model_id(model)

    settings: dict = {
        "prompt": prompt,
        "num_steps": int(steps),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "prompt_adherence": float(prompt_adherence),
    }
    if name:
        settings["name"] = name
    machine = _machine(gpu_type, gpu_count)
    if machine:
        settings["machine"] = machine

    lora_details = None
    if lora_id:
        detail: dict = {"id": lora_id, "weight": float(weight)}
        if trigger_word:
            detail["trigger_word"] = trigger_word
        lora_details = [detail]

    resp = await ai.submit_inference(pid, model_id, settings, lora_details) or {}
    inference_id = str(resp.get("id") or "")
    if not inference_id:
        return {"ok": False, "error": f"Inference POST returned no id: {resp}"}

    return {
        "ok": True,
        "inference_id": inference_id,
        "short_id": resp.get("short_id"),
        "status": resp.get("latest_status"),
        "project_id": pid,
        "model_id": model_id,
        "prompt": prompt,
        "lora_id": lora_id or None,
        "trigger_word": trigger_word or None,
    }


def _classify(status: str) -> str:
    s = (status or "").strip().lower()
    if s in TERMINAL_OK:
        return "success"
    if s in TERMINAL_BAD:
        return "failed"
    return "running"


async def generation_status(inference_id: str) -> dict:
    """Where a generation has got to, with its cost so far.

    `state` is our three-way reading of `status`; the raw string is kept so a
    vocabulary we have not seen shows up in the output instead of being
    flattened away.
    """
    pid = await project_id()
    job = await ai.get_inference(pid, inference_id) or {}
    status = str(job.get("latest_status") or "")
    spend = job.get("spend") or {}
    history = [
        {"status": s.get("Status") or s.get("status"),
         "at": s.get("Timestamp") or s.get("timestamp")}
        for s in (job.get("statuses") or []) if isinstance(s, dict)
    ]
    return {
        "inference_id": inference_id,
        "status": status,
        "state": _classify(status),
        "history": history,
        "cost_usd": spend.get("total_cost"),
        "minutes": spend.get("total_minutes"),
        "prompt": (job.get("settings") or {}).get("prompt"),
    }


async def generation_images(inference_id: str) -> dict:
    """The generated images, as directly fetchable URLs.

    Each output asset carries its own presigned url — no separate signing round
    trip, unlike the render side's POST /jobs/{id}/downloads.
    """
    pid = await project_id()
    resp = await ai.inference_outputs(pid, inference_id) or {}
    images = []
    for row in (resp.get("data") or []):
        asset = (row.get("asset") if isinstance(row, dict) else None) or {}
        url = asset.get("presigned_url") or asset.get("url") or ""
        name = asset.get("file_name") or ""
        if not url:
            continue
        if name and not name.lower().endswith(IMAGE_EXTS):
            continue
        images.append({
            "url": url,
            "file_name": name,
            "asset_id": asset.get("asset_id") or asset.get("id"),
            "sequence_number": asset.get("sequence_number"),
            "mime_type": asset.get("mime_type"),
        })
    return {"inference_id": inference_id, "count": len(images), "images": images}


# --- MAM write-back --------------------------------------------------------

async def ingest_generated_images(
    inference_id: str,
    prompt: str,
    lora_item_id: str = "",
    base_model: str = "",
    created_by: str = "NearlyMe",
    collection: Optional[str] = lora_pipeline.LORA_OUTPUT_COLLECTION,
) -> dict:
    """Land this generation's images in Cantemo, each carrying its provenance.

    Intentionally a sibling of lora_pipeline.ingest_generated_images rather
    than a refactor of it: that function is on the frozen IBC path, and the
    only thing the two share is the per-image loop. Two differences justify
    the copy — images arrive as presigned URLs from a different endpoint, and
    lora_item_id is OPTIONAL here, because a base-model-only generation has no
    LoRA to relate to and should still land with its prompt recorded.
    """
    found = await generation_images(inference_id)
    images = found["images"]
    if not images:
        status = await generation_status(inference_id)
        return {
            "ok": False,
            "inference_id": inference_id,
            "error": f'Generation {inference_id} produced no images (status: {status["status"] or "unknown"}).',
            "status": status,
        }

    identity = {"label": "", "trigger_word": ""}
    if lora_item_id:
        identity = await lora_pipeline.lora_identity(lora_item_id)

    created: list[str] = []
    skipped_report: list[str] = []
    for idx, img in enumerate(images, start=1):
        item = await cantemo.create_placeholder(title=f"{prompt[:60]} ({idx})")
        item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
        if not item_id:
            continue
        # tags=<profile> is what makes Cantemo build the poster; notranscode
        # alone does nothing, because the poster is produced BY the transcode.
        await cantemo.import_uri(item_id, img["url"], notranscode=False,
                                 tags=lora_pipeline.TRANSCODE_TAGS or None)

        values = {
            "provenance_kind": "generated_image",
            "prompt": prompt,
            "base_model": base_model or "conductor-for-ai",
            "job_id": inference_id,
            "created_by": created_by,
        }
        if identity["label"]:
            values["label"] = identity["label"]
        if identity["trigger_word"]:
            values["trigger_word"] = identity["trigger_word"]
        if lora_item_id:
            values["source_asset_ids"] = lora_item_id

        prov_fields, prov_skipped = await lora_pipeline._writable_provenance_fields(values)
        skipped_report = prov_skipped
        await cantemo.set_metadata(item_id, prov_fields, group_name=lora_pipeline.PROVENANCE_GROUP)

        if lora_item_id:
            try:
                await cantemo.create_relation(item_id, lora_item_id,
                                              relation_type=lora_pipeline.REL_GENERATED_WITH)
            except Exception as exc:
                _logger.error("[conductor-ai] relation %s -> %s failed: %s", item_id, lora_item_id, exc)
        created.append(item_id)

    filed = await lora_pipeline._file_into(collection, created)
    return {
        "ok": True,
        "inference_id": inference_id,
        "items": created,
        "count": len(created),
        "filed": filed,
        "lora": {"item_id": lora_item_id, **identity} if lora_item_id else None,
        "provenance_skipped": skipped_report,
    }


# --- training (client-complete, not yet proven) -----------------------------

async def submit_lora_training(
    dataset_id: str,
    lora_name: str,
    model: str = "",
    trigger_word: str = "sks",
    num_epochs: int = 10,
    batch_size: int = 1,
    network_dimension: int = 32,
    network_alpha: int = 16,
    num_repeats: int = 10,
    learning_rate: str = "1e-4",
    seed: int = DEFAULT_SEED,
    sample_prompts: Optional[list[str]] = None,
    sample_every_n_epochs: int = 1,
    max_runtime_minutes: int = 60,
    gpu_type: str = "",
    gpu_count: int = 0,
) -> dict:
    """Train a LoRA through this API against an already-populated dataset.

    The dataset half (upload images -> asset ids -> dataset) is the part the
    spec does not document a request body for, so it is left to run_ai_probe.py
    to settle rather than guessed at in a code path that spends GPU money.
    Training itself is a documented shape, so it is written out here ready.
    """
    pid = await project_id()
    model_id = await resolve_model_id(model)
    settings: dict = {
        "lora_name": lora_name,
        "trigger_word": trigger_word,
        "num_epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "network_dimension": int(network_dimension),
        "network_alpha": int(network_alpha),
        "num_repeats": int(num_repeats),
        "learning_rate": learning_rate,
        "seed": int(seed),
        "sample_every_n_epochs": int(sample_every_n_epochs),
        "max_runtime_minutes": int(max_runtime_minutes),
    }
    if sample_prompts:
        settings["prompts"] = sample_prompts
    machine = _machine(gpu_type, gpu_count)
    if machine:
        settings["machine"] = machine

    resp = await ai.submit_training(pid, [dataset_id], model_id, settings) or {}
    training_id = str(resp.get("id") or "")
    if not training_id:
        return {"ok": False, "error": f"Training POST returned no id: {resp}"}
    return {"ok": True, "training_id": training_id, "short_id": resp.get("short_id"),
            "status": resp.get("latest_status"), "project_id": pid, "model_id": model_id}


async def training_status(training_id: str) -> dict:
    pid = await project_id()
    job = await ai.get_training(pid, training_id) or {}
    status = str(job.get("latest_status") or "")
    spend = job.get("spend") or {}
    return {
        "training_id": training_id,
        "status": status,
        "state": _classify(status),
        "epochs_completed": job.get("num_epochs_completed"),
        "cost_usd": spend.get("total_cost"),
        "minutes": spend.get("total_minutes"),
    }


async def training_epochs(training_id: str) -> dict:
    """Every checkpoint so far, with the sample image each one rendered."""
    pid = await project_id()
    resp = await ai.list_epochs(pid, training_id)
    rows = resp if isinstance(resp, list) else (resp or {}).get("data") or []
    epochs = [
        {
            "epoch": r.get("epoch"),
            "file_name": r.get("file_name"),
            "url": r.get("presigned_url") or r.get("url"),
            "asset_id": r.get("asset_id") or r.get("id"),
        }
        for r in rows if isinstance(r, dict)
    ]
    return {"training_id": training_id, "count": len(epochs), "epochs": epochs}
