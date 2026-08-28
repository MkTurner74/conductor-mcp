"""
Conductor MCP Server
Exposes Conductor by CoreWeave job management as MCP tools.
Any MCP-compatible AI agent can submit and manage render jobs via this server.
"""

import json
import os
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import asyncio

import cantemo_client as cantemo
import conductor_client as conductor
import conductor_render
import lora_pipeline

# Stateless mode (no in-memory MCP sessions) is required on serverless hosts
# (Vercel) where each request may hit a fresh instance. Local uvicorn keeps
# stateful sessions unless STATELESS_HTTP is set.
_STATELESS = bool(os.getenv("STATELESS_HTTP") or os.getenv("VERCEL"))

mcp = FastMCP("Conductor", host="0.0.0.0", streamable_http_path="/", stateless_http=_STATELESS)


class BearerAuthMiddleware:
    """Reject requests without the shared bearer token when MCP_AUTH_TOKEN is set.

    The hosted instance fronts Mark's Conductor account (job submission spends
    real money), so it must never run open on the public internet. Callers
    (the Samsyn app's /api/conductor route, the navigator dev proxy) hold the
    token server-side; it never reaches a browser.
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.token:
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if headers.get("authorization") != f"Bearer {self.token}":
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"error": "unauthorized"}',
                })
                return
        await self.app(scope, receive, send)


def build_app():
    """ASGI app with optional bearer auth — used by uvicorn and the Vercel entry."""
    app = mcp.streamable_http_app()
    token = os.getenv("MCP_AUTH_TOKEN", "").strip()
    return BearerAuthMiddleware(app, token) if token else app


@mcp.tool()
async def list_instance_types() -> str:
    """
    List all available hardware instance types on Conductor.
    Returns machine names, CPU/GPU specs, and indicative cost tiers.
    Use this before submitting a job to pick the right instance_type.
    """
    data = await conductor.list_instance_types()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_projects() -> str:
    """
    List all Conductor projects for the authenticated account.
    Use this to find valid project names before submitting a job.
    """
    data = await conductor.list_projects()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_software_packages() -> str:
    """
    List all available software packages on Conductor — Maya, Blender, Houdini, Nuke, Cinema 4D, etc.
    Returns package IDs and version info. Use package IDs when submitting a job.
    """
    data = await conductor.list_software_packages()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_jobs(job_id_start: int = None, job_id_end: int = None) -> str:
    """
    List render jobs on the account. Optionally filter by job ID range.

    Args:
        job_id_start: Start of job ID range to filter (optional)
        job_id_end:   End of job ID range to filter (optional)

    Returns status, progress, cost, and metadata for each job.
    """
    data = await conductor.list_jobs(job_id_start, job_id_end)
    return json.dumps(data, indent=2)


@mcp.tool()
async def submit_render_job(
    job_title: Annotated[str, Field(description="Display name shown in the Conductor dashboard")],
    project: Annotated[str, Field(description="Conductor project to run under — pick from list_projects")],
    instance_type: Annotated[str, Field(description="Machine type — pick from list_instance_types. All types run on CoreWeave cloud and any listed type is valid; GPU types name their GPU model")],
    software_package_ids: Annotated[list[str], Field(description="Software package IDs to load on the render node — pick from list_software_packages")],
    tasks: Annotated[list[dict], Field(description='Task list. Each task: {"command": "<render command>", "frames": "1-10"}')],
    output_path: Annotated[str, Field(description="Directory the render command writes into. Conductor syncs it back to its storage; outputs are then retrievable as signed URLs via get_job_outputs")],
    priority: Annotated[int, Field(description="1–10, higher runs sooner. Default 5")] = 5,
    preemptible: Annotated[bool, Field(description="true = discounted spot capacity that can be interrupted and retried (fine for most renders); false = uninterrupted, costs more. Default true")] = True,
    notify: Annotated[Optional[list[str]], Field(description="Email addresses to notify when the job finishes")] = None,
    metadata: Annotated[Optional[dict], Field(description="Custom key-value tags for tracking/reporting")] = None,
) -> str:
    """
    Submit a render job to Conductor.

    Args:
        job_title:             Display name shown in the Conductor dashboard
        project:               Project name — use list_projects() to find valid names
        instance_type:         Machine type e.g. 'standard', 'highcpu', 'highmem'
                               Use list_instance_types() to see all options
        software_package_ids:  List of software package IDs to load
                               Use list_software_packages() to find IDs
        tasks:                 List of task dicts. Each task must have:
                                 - "command": the render command to run (str)
                                 - "frames": frame range e.g. "1-100" (str, optional)
                               Example: [{"command": "vray -sceneFile=/path/scene.vrscene", "frames": "1-50"}]
        output_path:           Where rendered output should be written
        priority:              Job priority 1–10. Higher = runs sooner. Default 5.
        preemptible:           Use preemptible/spot instances to reduce cost. Default True.
        notify:                Email addresses to notify on completion (optional)
        metadata:              Custom key-value pairs for reporting/tracking (optional)

    Returns job ID and submission status.
    Note: This tool assumes scene files are already uploaded to Conductor storage.
    For local file upload, use the Conductor desktop client or CLI before submitting.
    """
    payload: dict = {
        "job_title": job_title,
        "project": project,
        "machine_flavor": instance_type,
        "tasks_data": tasks,
        "output_path": output_path,
        "priority": priority,
        "preemptible": preemptible,
        "software_package_ids": software_package_ids,
    }
    if notify:
        payload["notify"] = notify
    if metadata:
        payload["metadata"] = metadata

    data = await conductor.submit_job(payload)
    return json.dumps(data, indent=2)


@mcp.tool()
async def submit_houdini_render(
    output_path: Annotated[str, Field(description="Directory the render writes PNG frames into (e.g. /my_renders/samsyn_render). Retrieve results via get_job_outputs")],
    project: Annotated[str, Field(description="Conductor project (default TestProject) — see list_projects")] = "TestProject",
    instance_type: Annotated[str, Field(description="CoreWeave machine type (default cw-xeonv3-32) — see list_instance_types")] = "cw-xeonv3-32",
    frame_start: Annotated[int, Field(description="First frame (default 1)")] = 1,
    frame_end: Annotated[int, Field(description="Last frame (default 24). Keep small — real GPU/CPU minutes cost money")] = 24,
    res_x: Annotated[int, Field(description="Width in px (default 1280)")] = 1280,
    res_y: Annotated[int, Field(description="Height in px (default 720)")] = 720,
    preemptible: Annotated[bool, Field(description="Spot capacity (cheaper, retried). Default true")] = True,
) -> str:
    """
    Submit a self-contained Houdini/Karma render to Conductor (CoreWeave) — no
    file uploads. Builds a spinning 3D object scene at runtime and renders a PNG
    frame sequence. Uses the ciocore SDK (the working submission path) and
    auto-resolves the latest Houdini 21 package.

    Renders FRAMES ONLY. To make a video, retrieve the frames with
    get_job_outputs (signed URLs) and assemble them downstream (Botverse
    assemble_sequence, or client-side) — CoreWeave render nodes have no ffmpeg
    and no egress, so in-node assembly is not available.

    Returns the job id (jid) + output path. Poll get_render_status(jid); when
    done, call get_job_outputs(jid).
    """
    # ciocore Submit is synchronous — run off the event loop.
    result = await asyncio.to_thread(
        conductor_render.submit_houdini_render,
        project=project, instance_type=instance_type, output_path=output_path,
        frame_start=frame_start, frame_end=frame_end, res_x=res_x, res_y=res_y,
        preemptible=preemptible,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_render_status(jid: str) -> str:
    """
    Status of a render job by its job id (jid, e.g. "00003" or "3").

    Use this instead of list_jobs for a single job: the list_jobs job-id range
    filter does not actually filter, so this matches by jid explicitly. Returns
    status (pending/running/success/failed), per-state task counts, and title.
    When status is success (or failed-with-frames), call get_job_outputs(jid).
    """
    data = await conductor_render.render_status(jid)
    return json.dumps(data, indent=2)


@mcp.tool()
async def kill_jobs(job_ids: list[int], action: str = "kill") -> str:
    """
    Cancel or hold one or more render jobs.

    Args:
        job_ids: List of integer job IDs to act on
        action:  What to do — 'kill' to cancel, 'hold' to pause (default: 'kill')

    Returns updated status for each affected job.
    """
    data = await conductor.kill_jobs(job_ids, action)
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_job_outputs(job_id: str, task_ids: list[str] = None) -> str:
    """
    List the rendered output files of a completed job, each with a signed
    download URL valid for direct ingestion by another service — no local
    download needed. Use after a job completes to hand outputs to a
    transcode step, an upload, or a browser download.

    Args:
        job_id:   The Conductor job ID (from list_jobs or submit_render_job)
        task_ids: Optional list of task IDs to restrict to (default: all tasks)

    Returns {"job_id": ..., "downloads": [tasks]} where each task carries a
    "files" list; every file has a signed "url" plus original path, size, and md5.
    Signed URLs expire — fetch them again if a downstream step runs much later.
    """
    data = await conductor.get_job_outputs(job_id, task_ids)
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_task_log(job_id: str, task_id: str) -> str:
    """
    Retrieve the log output for a specific task within a render job.
    Useful for diagnosing errors on failed tasks.

    Args:
        job_id:  The Conductor job ID (from list_jobs or submit_render_job response)
        task_id: The task ID within the job
    """
    data = await conductor.get_task_log(job_id, task_id)
    return json.dumps(data, indent=2) if isinstance(data, dict) else str(data)


# --- Cantemo MAM <-> LoRA round trip -------------------------------------
#
# The CoreWeave IBC demo. These sit alongside the render tools rather than in
# their own server because they are the same plumbing: Conductor jobs, submitted
# and polled the same way. Only the job type is new.
#
# Every tool here degrades to a clear message when Cantemo is not configured,
# so the render half of the server still works without MAM credentials.


def _cantemo_guard() -> Optional[str]:
    if not cantemo.configured():
        return json.dumps({"error": "Cantemo not configured — set CANTEMO_URL and CANTEMO_API_TOKEN"})
    return None


@mcp.tool()
async def cantemo_search_assets(
    query: Annotated[str, Field(description="Free-text search. Empty string matches everything")] = "",
    media_type: Annotated[Optional[str], Field(description="Filter to one media type: image, video, audio")] = None,
    limit: Annotated[int, Field(description="Max results. Default 25")] = 25,
) -> str:
    """
    Search assets in the Cantemo Portal MAM.

    Use this to find the items to train a LoRA on. Returns item ids (VX-NNNN),
    titles and media types — the ids are what submit_lora_training expects.
    """
    if err := _cantemo_guard():
        return err
    terms = [{"name": "mediaType", "value": media_type}] if media_type else None
    data = await cantemo.search(
        query=query, fields=["id", "title", "mediaType", "originalFilename"], terms=terms, limit=limit
    )
    results = [
        {
            "id": r.get("id"),
            "title": (r.get("title") or [None])[0],
            "mediaType": (r.get("mediaType") or [None])[0],
        }
        for r in (data or {}).get("results", [])
    ]
    return json.dumps({"hits": (data or {}).get("hits"), "results": results}, indent=2)


@mcp.tool()
async def cantemo_collection_items(
    collection_name: Annotated[str, Field(description='Name of the Cantemo collection to read, e.g. "Train LoRA"')],
    media_type: Annotated[Optional[str], Field(description="Filter to one media type: image, video, audio. Default image")] = "image",
) -> str:
    """
    List the items a user has curated into a named Cantemo collection.

    This is how the round trip gets STARTED from the MAM. Cantemo offers no
    webhook and no way to add a button without a plugin Codemill would have to
    install, so instead of pushing, we read: someone drags the assets they want
    trained into a collection, and the workflow picks them up from there. An
    ordinary MAM gesture, no custom UI.

    Returns {items: [...]} — the ids feed straight into submit_lora_training.
    """
    if err := _cantemo_guard():
        return err
    try:
        coll_id = await cantemo.find_collection(collection_name)
        if not coll_id:
            return json.dumps({"error": f'No collection named "{collection_name}"'}, indent=2)
        found = await cantemo.collection_items(coll_id, media_type=media_type or None)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(
        {
            "collection": collection_name,
            "collection_id": coll_id,
            "count": len(found),
            "items": [f["id"] for f in found],
            "detail": found,
        },
        indent=2,
    )


@mcp.tool()
async def cantemo_add_to_collection(
    collection_name: Annotated[str, Field(description='Collection to file the items into, e.g. "LoRA Output". Created if absent')],
    item_ids: Annotated[list[str], Field(description="Cantemo item ids to add")],
) -> str:
    """
    File items into a named Cantemo collection, creating it if it does not exist.

    Used as the closing step of a generation run so the new images land somewhere
    a person actually looks, rather than only existing as loose items. Adds
    without moving, so an item stays in any collection it is already in.
    """
    if err := _cantemo_guard():
        return err
    if not item_ids:
        return json.dumps({"error": "No item_ids to add"}, indent=2)
    try:
        coll_id = await cantemo.find_collection(collection_name)
        created = False
        if not coll_id:
            coll_id = await cantemo.create_collection(collection_name)
            created = True
        if not coll_id:
            return json.dumps({"error": f'Could not find or create "{collection_name}"'}, indent=2)
        task = await cantemo.add_to_collection(coll_id, item_ids)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(
        {
            "ok": True,
            "collection": collection_name,
            "collection_id": coll_id,
            "created_collection": created,
            "added": len(item_ids),
            "items": item_ids,
            # Cantemo performs the add in the background, so the collection may
            # read empty for a few seconds afterwards.
            "task": task,
        },
        indent=2,
    )


@mcp.tool()
async def cantemo_stamp_training_sources(
    source_item_ids: Annotated[list[str], Field(description="The Cantemo items that were trained on")],
    label: Annotated[str, Field(description="Name of the LoRA they produced")],
    job_id: Annotated[str, Field(description="Conductor training job id")],
    base_model: Annotated[str, Field(description="Base model product")] = "sdxl1-kohya",
    trigger_word: Annotated[str, Field(description="Trigger token the LoRA was trained with")] = "",
) -> str:
    """
    Write provenance onto the training images themselves.

    Without this, a source still shows "There is no metadata for this item yet"
    in the MAM even though it helped train a model. Provenance should read both
    ways: the LoRA says what it came from, and each contributing image says what
    it went into.

    Uses the same eight fields with prov_kind = training_source.
    """
    if err := _cantemo_guard():
        return err
    try:
        result = await lora_pipeline.stamp_source_assets(
            source_item_ids=source_item_ids, label=label, job_id=job_id,
            base_model=base_model, trigger_word=trigger_word,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def cantemo_remove_from_collection(
    collection_name: Annotated[str, Field(description='Collection to remove from, e.g. "Train LoRA"')],
    item_ids: Annotated[list[str], Field(description="Cantemo item ids to remove")],
) -> str:
    """
    Take items out of a collection without deleting the items.

    The collection IS the training set, so curating it matters: a LoRA trained
    on a folder holding two unrelated subjects learns a muddle of both.
    """
    if err := _cantemo_guard():
        return err
    try:
        coll_id = await cantemo.find_collection(collection_name)
        if not coll_id:
            return json.dumps({"error": f'No collection named "{collection_name}"'}, indent=2)
        await cantemo.remove_from_collection(coll_id, item_ids)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(
        {"ok": True, "collection": collection_name, "collection_id": coll_id,
         "removed": len(item_ids), "items": item_ids}, indent=2,
    )


@mcp.tool()
async def cantemo_get_asset(
    item_id: Annotated[str, Field(description="Cantemo item id, e.g. VX-4153")],
) -> str:
    """Full detail for one Cantemo item, including its shapes and provenance relations."""
    if err := _cantemo_guard():
        return err
    item, formats, relations = await asyncio.gather(
        cantemo.get_item(item_id),
        cantemo.get_formats(item_id),
        cantemo.get_relations(item_id),
        return_exceptions=True,
    )
    def safe(v):
        return {"error": str(v)} if isinstance(v, Exception) else v
    return json.dumps({"item": safe(item), "formats": safe(formats), "relations": safe(relations)}, indent=2, default=str)


@mcp.tool()
async def submit_lora_training(
    item_ids: Annotated[list[str], Field(description="Cantemo item ids to train on — get these from cantemo_search_assets")],
    label: Annotated[str, Field(description='Human-readable name, e.g. "Aston Martin F1 — Livery v1"')],
    trigger_word: Annotated[str, Field(description="Token that invokes the trained concept in prompts. Default 'sks'")] = "sks",
    class_word: Annotated[str, Field(description="What kind of thing this is: style, livery, character, object")] = "style",
    model: Annotated[str, Field(description="Base model: sdxl, flux-schnell, flux-dev, sd35-large, sd35-medium, sd3-medium")] = "sdxl",
    epochs: Annotated[int, Field(description="Training epochs. Default 10")] = 10,
    dry_run: Annotated[bool, Field(description="True (default) builds and validates the job WITHOUT spending GPU money. Set false only on explicit instruction")] = True,
) -> str:
    """
    Train a LoRA on CoreWeave GPUs from assets selected in the Cantemo MAM.

    Downloads the chosen images, lays them out as a kohya dataset, and submits a
    training job to Conductor using the kohya package whose base model already
    lives on the node (no egress needed).

    DEFAULTS TO A DRY RUN. A real submission spends money on Mark's Conductor
    account — only pass dry_run=false when explicitly told to.

    Non-image items in the selection are skipped and reported, not fatal.
    """
    if err := _cantemo_guard():
        return err
    workdir = os.path.join(
        os.getenv("LORA_WORKDIR", os.path.join(os.path.expanduser("~"), ".conductor-lora")),
        "".join(ch if ch.isalnum() else "-" for ch in label).strip("-").lower(),
    )
    try:
        result = await lora_pipeline.submit_training(
            dry_run=dry_run,
            item_ids=item_ids,
            label=label,
            workdir=workdir,
            model=model,
            trigger_word=trigger_word,
            class_word=class_word,
            epochs=epochs,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def ingest_lora_to_mam(
    job_id: Annotated[str, Field(description="Conductor job id of the finished training job")],
    label: Annotated[str, Field(description="Name for the LoRA item in the MAM")],
    source_item_ids: Annotated[list[str], Field(description="The Cantemo items it was trained on — recorded as provenance relations")],
    trigger_word: Annotated[str, Field(description="The trigger token used during training")] = "sks",
    base_model: Annotated[str, Field(description="Base model product it was trained against")] = "sdxl1-kohya",
    created_by: Annotated[str, Field(description="Who ran the training")] = "NearlyMe",
) -> str:
    """
    Land a finished LoRA back in the Cantemo MAM with full provenance.

    Creates a new item, imports the .safetensors from Conductor's signed output
    URL, writes the provenance metadata, and creates a relation edge to every
    source asset so the MAM itself can answer "what was this trained on".
    """
    if err := _cantemo_guard():
        return err
    try:
        result = await lora_pipeline.ingest_lora_to_cantemo(
            job_id=job_id, label=label, source_item_ids=source_item_ids,
            base_model=base_model, trigger_word=trigger_word, created_by=created_by,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def submit_lora_inference(
    lora_item_id: Annotated[str, Field(description="Cantemo item id of the LoRA to generate with — the item created by ingest_lora_to_mam")],
    prompt: Annotated[str, Field(description="What to generate. Include the LoRA's trigger word to invoke the trained concept")],
    count: Annotated[int, Field(description="Images to generate. Default 4")] = 4,
    steps: Annotated[int, Field(description="Sampling steps. Default 30")] = 30,
    seed: Annotated[int, Field(description="Random seed for reproducibility. Default 42")] = 42,
    model: Annotated[str, Field(description="Base model the LoRA was trained against. Default sdxl")] = "sdxl",
    dry_run: Annotated[bool, Field(description="True (default) builds the job WITHOUT spending GPU money. Set false only on explicit instruction")] = True,
) -> str:
    """
    Generate images from a LoRA held in the Cantemo MAM.

    Fetches the LoRA's weights back out of the MAM, uploads them to a CoreWeave
    GPU node, and runs generation against the base model already on that node.

    DEFAULTS TO A DRY RUN — a real submission spends money.
    """
    if err := _cantemo_guard():
        return err
    workdir = os.path.join(
        os.getenv("LORA_WORKDIR", os.path.join(os.path.expanduser("~"), ".conductor-lora")),
        "infer", lora_item_id,
    )
    try:
        result = await lora_pipeline.submit_inference(
            dry_run=dry_run, lora_item_id=lora_item_id, prompt=prompt,
            workdir=workdir, count=count, steps=steps, seed=seed, model=model,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def ingest_generated_to_mam(
    job_id: Annotated[str, Field(description="Conductor job id of the finished inference job")],
    lora_item_id: Annotated[str, Field(description="The LoRA item these were generated with")],
    prompt: Annotated[str, Field(description="The prompt used — recorded on every generated item")],
    base_model: Annotated[str, Field(description="Base model product")] = "sdxl1-kohya",
    created_by: Annotated[str, Field(description="Who ran the generation")] = "NearlyMe",
) -> str:
    """
    Land generated images back in the Cantemo MAM, each carrying its prompt and
    a relation edge back to the LoRA that produced it — closing the round trip.
    """
    if err := _cantemo_guard():
        return err
    try:
        result = await lora_pipeline.ingest_generated_images(
            job_id=job_id, lora_item_id=lora_item_id, prompt=prompt,
            base_model=base_model, created_by=created_by,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def cantemo_provenance_plan() -> str:
    """
    Check whether the "AI Provenance" metadata group exists on the Portal, and
    report the fields the round trip needs.

    Field creation is not exposed on Cantemo's v2 API — it goes through the
    Portal admin UI or Vidispine — so this reports the plan rather than applying it.
    """
    if err := _cantemo_guard():
        return err
    return json.dumps(await lora_pipeline.bootstrap_provenance_group(), indent=2, default=str)


@mcp.tool()
async def kohya_package_info(
    model: Annotated[str, Field(description="Base model key: sdxl, flux-schnell, flux-dev, sd35-large, sd35-medium, sd3-medium")] = "sdxl",
) -> str:
    """
    Show which Conductor packages a LoRA job would load and where the base model
    sits on the node. Read-only — useful for confirming the training environment
    before spending anything.
    """
    try:
        return json.dumps(await lora_pipeline.resolve_packages(model), indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        # Use Streamable HTTP (POST /mcp) — the modern MCP transport expected by claude.ai.
        uvicorn.run(build_app(), host="0.0.0.0", port=port)
    else:
        mcp.run()  # stdio — used by Claude Desktop and local agents
