"""
Conductor REST API client.
Handles auth (API key → bearer token) and wraps all endpoints used by the MCP tools.
"""

import json
import os
from typing import Any, Optional

import httpx

CONDUCTOR_API_URL = os.getenv("CONDUCTOR_API_URL", "https://api.conductortech.com")

# The env var can be either:
#   - Raw JSON contents of the downloaded key file: {"api_key": "..."}
#   - Just the key string itself
_raw = os.getenv("CONDUCTOR_API_KEY", "")
try:
    CONDUCTOR_API_KEY = json.loads(_raw).get("api_key", _raw)
except (json.JSONDecodeError, AttributeError, TypeError):
    CONDUCTOR_API_KEY = _raw

_bearer_token: Optional[str] = None


async def _get_bearer_token() -> str:
    global _bearer_token
    if _bearer_token:
        return _bearer_token
    async with httpx.AsyncClient(timeout=10.0) as client:
        # NOTE: verify this auth endpoint against your account if it returns 404
        resp = await client.post(
            f"{CONDUCTOR_API_URL}/api/v1/api_key/auth",
            json={"api_key": CONDUCTOR_API_KEY},
        )
        resp.raise_for_status()
        _bearer_token = resp.json()["token"]
        return _bearer_token


async def _get(path: str, params: Optional[dict] = None) -> Any:
    token = await _get_bearer_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{CONDUCTOR_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, payload: dict) -> Any:
    token = await _get_bearer_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{CONDUCTOR_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def _put(path: str, payload: dict) -> Any:
    token = await _get_bearer_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{CONDUCTOR_API_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# --- Public API wrappers ---

async def list_instance_types() -> Any:
    return await _get("/api/v1/instance-types")


async def list_projects() -> Any:
    return await _get("/api/v1/projects/")


async def list_software_packages() -> Any:
    return await _get("/api/v1/ee/packages")


async def list_jobs(job_id_start: Optional[int] = None, job_id_end: Optional[int] = None) -> Any:
    params: dict = {}
    if job_id_start is not None:
        params["job_id_start"] = job_id_start
    if job_id_end is not None:
        params["job_id_end"] = job_id_end
    return await _get("/api/v1/jobs", params=params)


async def submit_job(payload: dict) -> Any:
    # NOTE: if this returns 404, the submit endpoint may be /api/v1/submit or /api/v1/job
    return await _post("/api/v1/jobs", payload)


async def kill_jobs(job_ids: list[int], action: str = "kill") -> Any:
    return await _put("/jobs_multi", {"job_ids": job_ids, "status": action})


async def get_task_log(job_id: str, task_id: str) -> Any:
    return await _get("/get_log_file", params={"job_id": job_id, "task_id": task_id})
