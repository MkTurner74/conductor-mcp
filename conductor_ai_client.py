"""
Conductor for AI — REST client.

This is a DIFFERENT product from the render-job system the rest of this server
talks to, not a new endpoint on the same one:

  * different host      — backend.conductortech.com, not dashboard.conductortech.com
  * different auth      — a plain API key in the Authorization header, NOT the
                          RSA-service-account -> OAuth bearer dance in
                          conductor_client.py. Proven 2026-09-03: a valid
                          bearer token from the old flow returns 200 on the old
                          API and a clean 401 here, raw and Bearer-prefixed
                          alike. The key is genuinely separate.
  * different model     — first-class base-models / lora-trainings / inferences
                          resources, not a generic job queue. Models stay
                          resident, so inference should not pay the per-job
                          cold start the kohya-bash path pays.

Nothing here touches lora_pipeline.py's proven path. That path is locked for
IBC; this is additive and parallel, and can be abandoned without unpicking
anything.

Configuration (all read at call time, so a key can be dropped in without an
import-order surprise):

  CONDUCTOR_AI_API_KEY     the key issued for THIS API. Required.
  CONDUCTOR_AI_PROJECT_ID  uuid of the Conductor-for-AI project. Optional if
                           CONDUCTOR_AI_PROJECT names one instead.
  CONDUCTOR_AI_PROJECT     project NAME, resolved to an id via GET /projects.
  CONDUCTOR_AI_API_URL     override the host (default backend.conductortech.com).
  CONDUCTOR_AI_AUTH_STYLE  "raw" | "bearer". Unset = try raw, fall back to
                           bearer on a 401 and remember which one worked.

Spec of record: https://backend.conductortech.com/core/doc/openapi.json
(Core API v1.0.0, 79 paths, security scheme apiKey in header "Authorization").
"""

import logging
import os
from typing import Any, Optional

import httpx

_logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://backend.conductortech.com"

# Which Authorization style the server actually accepted, remembered for the
# process once proven. The spec says apiKey-in-header, which conventionally
# means the bare key, but plenty of "apiKey" schemes still want "Bearer ".
# Guessing wrong costs a 401 that looks like a bad key, so try and remember.
_auth_style: Optional[str] = None


class ConductorAIError(RuntimeError):
    """An HTTP failure with the response body kept.

    Deliberately verbose: a status code alone sent three LoRA jobs' worth of
    debugging down the wrong path once already. The body is where this API puts
    messages like "Job is missing required attributes".
    """

    def __init__(self, method: str, path: str, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:600]}")


def api_url() -> str:
    val = (os.getenv("CONDUCTOR_AI_API_URL", "") or "").strip() or DEFAULT_API_URL
    return val if "://" in val else f"https://{val}"


def api_key() -> str:
    # .strip() is not cosmetic: a trailing \r from a Windows-generated env value
    # made Vercel's edge 400 an entire deployment's requests once.
    return (os.getenv("CONDUCTOR_AI_API_KEY", "") or "").strip()


def configured() -> bool:
    return bool(api_key())


def _require_key() -> str:
    key = api_key()
    if not key:
        raise ConductorAIError(
            "GET", "/core/v1", 0,
            "CONDUCTOR_AI_API_KEY is not set. Conductor for AI needs its own key — "
            "the render API's service-account key does not authenticate here.",
        )
    return key


def _header(style: str, key: str) -> dict:
    return {"Authorization": key if style == "raw" else f"Bearer {key}"}


async def _request(method: str, path: str, *, params: Optional[dict] = None,
                   json_body: Optional[Any] = None, timeout: float = 60.0) -> Any:
    """One request, with the auth-style probe folded in.

    On the first 401 with an unpinned style, retry once with the other header
    shape and remember the winner. Every later call goes straight there.
    """
    global _auth_style
    key = _require_key()
    pinned = (os.getenv("CONDUCTOR_AI_AUTH_STYLE", "") or "").strip().lower()
    styles = [pinned] if pinned in ("raw", "bearer") else (
        [_auth_style] if _auth_style else ["raw", "bearer"]
    )

    last: Optional[ConductorAIError] = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for style in styles:
            resp = await client.request(
                method, f"{api_url()}{path}",
                headers={**_header(style, key), "Accept": "application/json"},
                params=params or None, json=json_body,
            )
            if resp.status_code == 401 and len(styles) > 1:
                last = ConductorAIError(method, path, resp.status_code, resp.text)
                continue
            if resp.status_code >= 400:
                raise ConductorAIError(method, path, resp.status_code, resp.text)
            if _auth_style != style:
                _auth_style = style
                _logger.info("[conductor-ai] authenticating with the %s Authorization style", style)
            if resp.status_code == 204 or not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                return {"raw": resp.text}
    raise last or ConductorAIError(method, path, 401, "unauthorized")


# --- Projects --------------------------------------------------------------

async def list_projects(limit: int = 100) -> Any:
    return await _request("GET", "/core/v1/projects", params={"limit": limit})


async def get_project(project_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}")


async def create_project(name: str, description: str = "", owner: str = "") -> Any:
    body: dict = {"name": name, "description": description}
    if owner:
        body["owner"] = owner
    return await _request("POST", "/core/v1/projects", json_body=body)


async def resolve_project_id() -> str:
    """The project every other path is scoped under.

    Prefers the explicit id, falls back to matching CONDUCTOR_AI_PROJECT by
    name, and finally to the only project on the account if there is exactly
    one — the same shape the render side has (a lone "TestProject"), and one
    less thing to configure on a laptop.
    """
    explicit = (os.getenv("CONDUCTOR_AI_PROJECT_ID", "") or "").strip()
    if explicit:
        return explicit

    wanted = (os.getenv("CONDUCTOR_AI_PROJECT", "") or "").strip()
    rows = (await list_projects()).get("data") or []
    if wanted:
        for row in rows:
            if str(row.get("name", "")).lower() == wanted.lower():
                return str(row.get("id"))
        raise ConductorAIError("GET", "/core/v1/projects", 404,
                               f'No Conductor-for-AI project named "{wanted}". Found: '
                               + ", ".join(str(r.get("name")) for r in rows))
    if len(rows) == 1:
        return str(rows[0].get("id"))
    raise ConductorAIError("GET", "/core/v1/projects", 400,
                           "Set CONDUCTOR_AI_PROJECT_ID or CONDUCTOR_AI_PROJECT — the account has "
                           f"{len(rows)} projects: " + ", ".join(str(r.get('name')) for r in rows))


# --- Base models and LoRAs -------------------------------------------------

async def list_base_models(project_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/base-models")


async def get_base_model(project_id: str, model_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/base-models/{model_id}")


async def create_base_model(project_id: str, body: dict) -> Any:
    """Register a base model. Only needed if the account has none pre-registered.

    Body shape (BaseModelPostRequest): name, model_name, model_family,
    model_path, model_filename, engine_path, engine_version, encoders_path,
    env_path, command, t5_encoder_filename, publish, settings.
    """
    return await _request("POST", f"/core/v1/projects/{project_id}/base-models", json_body=body)


async def list_loras(project_id: str, model_id: str, limit: int = 100,
                     cursor: Optional[str] = None) -> Any:
    params: dict = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    return await _request(
        "GET", f"/core/v1/projects/{project_id}/base-models/{model_id}/lora-models", params=params)


async def get_lora(project_id: str, lora_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/lora-models/{lora_id}")


# --- Datasets and assets (training inputs) ---------------------------------

async def list_datasets(project_id: str, limit: int = 100) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/assets/datasets",
                          params={"limit": limit})


async def create_dataset(project_id: str, asset_ids: list[str], name: str = "") -> Any:
    """Create a dataset.

    CONFIRMED 2026-09-09 via run_ai_probe.py --dataset-probe: `asset_ids` is
    REQUIRED (server: "Field validation for 'AssetIDs' failed on the
    'required' tag"). There is no create-empty-then-attach two-step — the
    assets have to exist (via sign_uploads + complete_multipart) before this
    call, and their ids go in the same POST that creates the dataset.
    """
    if not asset_ids:
        raise ValueError("create_dataset needs at least one asset_id — upload the images first "
                         "(sign_uploads + complete_multipart), there is no empty-dataset-then-attach path.")
    body: dict = {"name": name, "asset_ids": asset_ids}
    return await _request("POST", f"/core/v1/projects/{project_id}/assets/dataset", json_body=body)


async def update_dataset(project_id: str, dataset_id: str, name: str = "",
                         asset_ids: Optional[list[str]] = None) -> Any:
    """Attach assets to a dataset. Same undocumented-body caveat as create."""
    body: dict = {}
    if name:
        body["name"] = name
    if asset_ids is not None:
        body["asset_ids"] = asset_ids
    return await _request("PUT", f"/core/v1/projects/{project_id}/assets/dataset/{dataset_id}",
                          json_body=body)


async def sign_uploads(project_id: str, files: list[dict], cloud: str = "",
                       type_metadata: Optional[dict] = None) -> Any:
    """Create asset records and get presigned upload URLs.

    files: [{file_name, file_path, file_extension, mime_type, size, hash}]
    Returns [{asset_id, s3_upload_id, part_size, parts:[{part_number, url}]}]
    — i.e. always multipart, so the caller PUTs each part and then calls
    complete_multipart with the ETags.
    """
    body: dict = {"upload_files": files}
    if cloud:
        body["cloud"] = cloud
    if type_metadata:
        body["type_metadata"] = type_metadata
    return await _request("POST", f"/core/v1/projects/{project_id}/sign/uploads", json_body=body)


async def complete_multipart(project_id: str, s3_upload_id: str, hash_: str,
                             completed_parts: list[dict], cloud: str = "") -> Any:
    body: dict = {"s3_upload_id": s3_upload_id, "hash": hash_, "completed_parts": completed_parts}
    if cloud:
        body["cloud"] = cloud
    return await _request("POST", f"/core/v1/projects/{project_id}/multipart/complete", json_body=body)


async def list_assets(project_id: str, limit: int = 100) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/assets", params={"limit": limit})


# --- LoRA training ---------------------------------------------------------

async def submit_training(project_id: str, dataset_ids: list[str], model_id: str,
                          settings: dict, parent_job_id: str = "",
                          inputs: Optional[dict] = None) -> Any:
    """
    dataset_ids/create_dataset is being retired (Conductor, 2026-09-10): submit
    directly with dataset_ids=[] and `inputs` as {asset_id: caption}. Kept as
    an optional param rather than replacing dataset_ids outright since
    Conductor said the field itself is not gone yet, just no longer required.
    """
    body: dict = {"dataset_ids": dataset_ids, "model_id": model_id, "settings": settings}
    if parent_job_id:
        body["parent_job_id"] = parent_job_id
    if inputs is not None:
        body["inputs"] = inputs
    return await _request("POST", f"/core/v1/projects/{project_id}/lora-trainings", json_body=body)


async def get_training(project_id: str, training_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/lora-trainings/{training_id}")


async def list_trainings(project_id: str, limit: int = 50) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/lora-trainings",
                          params={"limit": limit})


async def list_epochs(project_id: str, training_id: str, limit: int = 3,
                      order_by_asc: bool = True) -> Any:
    """Per-epoch checkpoints with their sample renders — watch it learn, and
    pick the epoch that looks right instead of taking whatever the last one
    produced. The render path has no equivalent.

    PAGINATED, default page size 3 (confirmed via the OpenAPI spec 2026-09-12
    — not documented in this function until a real 8-epoch job exposed it:
    the response has no `data`/cursor echoed back, just {"epochs": [...],
    "has_more": bool}). Ask for order_by_asc=False, limit=1 to get the most
    recent epoch in one call instead of paging through from the start.
    """
    params = {"limit": limit, "order_by_asc": str(order_by_asc).lower()}
    return await _request(
        "GET", f"/core/v1/projects/{project_id}/lora-trainings/{training_id}/epochs", params=params)


async def get_epoch(project_id: str, training_id: str, epoch_number: int) -> Any:
    return await _request(
        "GET", f"/core/v1/projects/{project_id}/lora-trainings/{training_id}/epochs/{epoch_number}")


async def training_logs(project_id: str, training_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/lora-trainings/{training_id}/logs")


async def training_action(project_id: str, training_id: str, action: str) -> Any:
    return await _request("POST", f"/core/v1/projects/{project_id}/lora-trainings/{training_id}/action",
                          json_body={"action": action})


# --- Inference -------------------------------------------------------------

async def submit_inference(project_id: str, model_id: str, settings: dict,
                           lora_details: Optional[list[dict]] = None) -> Any:
    """Start a generation.

    lora_details is a LIST — this API can stack several LoRAs in one call, each
    with its own trigger word and weight, which the kohya-bash path cannot do
    at all. Omit it entirely for a base-model-only generation (the cheapest
    proof that the API works, needing nothing uploaded).
    """
    body: dict = {"model_id": model_id, "settings": settings}
    if lora_details:
        body["lora_details"] = lora_details
    return await _request("POST", f"/core/v1/projects/{project_id}/inferences", json_body=body)


async def get_inference(project_id: str, inference_id: str) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/inferences/{inference_id}")


async def list_inferences(project_id: str, limit: int = 50) -> Any:
    return await _request("GET", f"/core/v1/projects/{project_id}/inferences", params={"limit": limit})


async def inference_outputs(project_id: str, inference_id: str) -> Any:
    """Generated images. Each asset carries `presigned_url`/`url` directly, so
    there is no separate signing step the way get_job_outputs needs one."""
    return await _request("GET", f"/core/v1/projects/{project_id}/inferences/{inference_id}/outputs")


async def inference_action(project_id: str, inference_id: str, action: str) -> Any:
    return await _request("POST", f"/core/v1/projects/{project_id}/inferences/{inference_id}/action",
                          json_body={"action": action})


async def job_statuses(project_id: str, job_id: str) -> Any:
    """Full status history for a training or inference job."""
    return await _request("GET", f"/core/v1/projects/{project_id}/statuses/{job_id}")


async def spends(limit: int = 50) -> Any:
    return await _request("GET", "/core/v1/billing/spends", params={"limit": limit})
