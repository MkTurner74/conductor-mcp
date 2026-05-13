"""
Conductor REST API client.
Handles auth (RSA service account key → bearer token) and wraps all endpoints.

The Conductor API key is a JSON file with two fields:
  { "client_id": "...", "private_key": "-----BEGIN PRIVATE KEY-----\n..." }

Set CONDUCTOR_API_KEY to the full JSON string, or CONDUCTOR_API_KEY_FILE to
the path of the downloaded JSON file.
"""

import json
import os
import time
from typing import Any, Optional

import httpx
import jwt  # PyJWT

CONDUCTOR_API_URL = os.getenv("CONDUCTOR_API_URL", "https://api.conductortech.com")

# Load key data from env var (JSON string) or file path
def _load_key() -> dict:
    raw = os.getenv("CONDUCTOR_API_KEY", "")
    if not raw:
        path = os.getenv("CONDUCTOR_API_KEY_FILE", "")
        if path:
            with open(path) as f:
                return json.load(f)
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}

_key_data = _load_key()
_bearer_token: Optional[str] = None
_token_expiry: float = 0


async def _get_bearer_token() -> str:
    global _bearer_token, _token_expiry
    if _bearer_token and time.time() < _token_expiry - 60:
        return _bearer_token

    client_id = _key_data.get("client_id", "")
    private_key = _key_data.get("private_key", "")
    if not client_id or not private_key:
        raise RuntimeError("CONDUCTOR_API_KEY not set or missing client_id/private_key fields")

    # Sign a short-lived JWT with the RSA private key
    now = int(time.time())
    signed_jwt = jwt.encode(
        {"iss": client_id, "sub": client_id, "aud": CONDUCTOR_API_URL, "iat": now, "exp": now + 3600},
        private_key,
        algorithm="RS256",
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{CONDUCTOR_API_URL}/api/v1/api_key/auth",
            json={"token": signed_jwt},
        )
        resp.raise_for_status()
        _bearer_token = resp.json()["token"]
        _token_expiry = now + 3600
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
