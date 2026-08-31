"""
Cantemo Portal REST client.

The MAM half of the CoreWeave IBC round trip: read the assets a user selected,
then write the trained LoRA (and anything generated from it) back as new items
carrying real provenance.

Auth is a single `auth-token` header against /API/v2/*. Verified against
Codemill's demo Portal 6.2.1 at cantemo6.codemill.se, which is reachable on the
open internet -- no VPN, no tunnel, unlike the Kitsu federation source.

Config:
  CANTEMO_URL        e.g. https://cantemo6.codemill.se  (no trailing slash needed)
  CANTEMO_API_TOKEN  the auth-token value
  CANTEMO_USER       Portal username -- UI login and Vidispine-level calls only
  CANTEMO_PASSWORD   ditto

Two things about this API that cost time to discover, both handled below:
  * Search is PUT, not GET or POST. Both of those return 405.
  * The media download endpoint answers 302 with a presigned S3 URL rather than
    streaming bytes, so the Location has to be read -- see resolve_download_url().
"""

import logging
import os
from typing import Any, Optional
from urllib.parse import quote

import httpx

_logger = logging.getLogger(__name__)

TIMEOUT = 60.0


def base_url() -> str:
    return os.getenv("CANTEMO_URL", "").rstrip("/")


def configured() -> bool:
    return bool(base_url() and os.getenv("CANTEMO_API_TOKEN"))


def _headers(accept: str = "application/json") -> dict:
    return {
        "auth-token": os.getenv("CANTEMO_API_TOKEN", ""),
        "Accept": accept,
    }


def _require() -> str:
    if not configured():
        raise RuntimeError("Cantemo not configured -- set CANTEMO_URL and CANTEMO_API_TOKEN")
    return base_url()


async def _request(method: str, path: str, **kw) -> Any:
    base = _require()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.request(method, f"{base}{path}", headers=_headers(), **kw)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()


# --- identity -------------------------------------------------------------

async def whoami() -> dict:
    """Who the token authenticates as, plus the portal_roles it carries."""
    return await _request("GET", "/API/v2/whoami/")


async def version() -> dict:
    return await _request("GET", "/API/v2/version/")


# --- read -----------------------------------------------------------------

async def search(
    query: str = "",
    fields: Optional[list[str]] = None,
    terms: Optional[list[dict]] = None,
    limit: int = 50,
) -> dict:
    """
    Search items. NB: PUT, not GET/POST.

    `terms` are ANDed with the always-present portal_deleted exclusion -- without
    that filter, deleted items come back in results.
    """
    body: dict = {
        "query": query,
        "filter": {
            "operator": "AND",
            "terms": [{"name": "portal_deleted", "missing": True}] + list(terms or []),
        },
    }
    if fields:
        body["fields"] = fields
    return await _request("PUT", f"/API/v2/search/?limit={int(limit)}", json=body)


async def find_collection(name: str) -> Optional[str]:
    """
    Resolve a collection's display name to its id.

    Autocomplete answers {"items": [{"VX-2556": "Clips"}]}, so the id is the key
    and the name is the value. Matched case-insensitively and exactly, because
    autocomplete is a prefix search and "Train" would otherwise pick up
    "Training Archive".
    """
    data = await _request("GET", f"/API/v2/collections/autocomplete/?query={quote(name)}")
    for entry in (data or {}).get("items", []):
        for coll_id, coll_name in entry.items():
            if str(coll_name).strip().lower() == name.strip().lower():
                return coll_id
    return None


async def collection_items(collection_id: str, media_type: Optional[str] = None) -> list[dict]:
    """
    Every item in a collection, optionally filtered to one media type.

    This is what lets the round trip be STARTED from the MAM without a Cantemo
    plugin: a user curates a collection the way they would for any other purpose,
    and the workflow reads it. No custom UI, no webhook -- neither of which
    Cantemo offers us.
    """
    data = await _request("GET", f"/API/v2/collections/{quote(collection_id)}/content/item/")
    items = []
    for obj in (data or {}).get("objects", []):
        kind = obj.get("mediaType")
        kind = kind[0] if isinstance(kind, list) and kind else kind
        if media_type and kind != media_type:
            continue
        title = obj.get("title")
        items.append({
            "id": obj.get("id"),
            "title": title[0] if isinstance(title, list) and title else title,
            "mediaType": kind,
        })
    return items


async def create_collection(name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """
    Create a collection. The field is `collection_name`, not `name` -- passing
    `name` returns a bare 500 with no hint.
    """
    body: dict = {"collection_name": name}
    if parent_id:
        body["parent_collection_id"] = parent_id
    data = await _request("POST", "/API/v2/collections/?refresh=true", json=body)
    return (data or {}).get("id")


async def ensure_collection(name: str, parent_name: Optional[str] = None) -> Optional[str]:
    """
    Find a collection by name, creating it -- and its parent -- if absent.

    Why outputs get their own collections rather than going back into the input
    one: "Train LoRA" is READ BY NAME at the start of every training run, so
    anything filed there becomes training data next time. Inputs and outputs
    have to be separate folders or the workflow feeds on its own results.

    The link back to the source assets is the relation edge, not folder
    membership -- which is the stronger claim anyway: the MAM answers "what was
    this trained on" by traversal, regardless of where someone filed it.
    """
    existing = await find_collection(name)
    if existing:
        return existing
    parent_id = None
    if parent_name:
        parent_id = await find_collection(parent_name) or await create_collection(parent_name)
    return await create_collection(name, parent_id=parent_id)


async def add_to_collection(collection_id: str, item_ids: list[str]) -> Optional[str]:
    """
    Add existing items to a collection without moving them.

    Two things about this endpoint cost time:
      * `selected_objects` must be REPEATED, one per id
        (?selected_objects=VX-1&selected_objects=VX-2). Comma-separating them
        returns 200 with a task id and then silently adds nothing.
      * The BODY names the target collection ([{"id": "VX-2601"}]); the query
        string names the assets. That is the opposite way round from what the
        parameter names suggest.
    Use this rather than PUT /content/, which MOVES items out of their current
    collection -- destructive on someone else's system.

    The work happens in the background, so a read straight afterwards may still
    show the collection empty. Returns the task id.
    """
    if not item_ids:
        return None
    qs = "&".join(f"selected_objects={quote(i)}" for i in item_ids)
    data = await _request(
        "POST", f"/API/v2/collections/content/?{qs}", json=[{"id": collection_id}]
    )
    return (data or {}).get("task")


async def remove_from_collection(collection_id: str, item_ids: list[str]) -> Any:
    """
    Take items back out of a collection, leaving the items themselves alone.

    Matters because the collection IS the training set: a LoRA trained on a
    folder holding two unrelated subjects learns a muddle of both. Curating the
    folder has to be as easy as filling it.

    Same repeated-`selected_objects` convention as add_to_collection, and the
    same background execution.
    """
    if not item_ids:
        return None
    qs = "&".join(f"selected_objects={quote(i)}" for i in item_ids)
    return await _request(
        "DELETE", f"/API/v2/collections/{quote(collection_id)}/content/item/?{qs}"
    )


async def get_item(item_id: str) -> dict:
    return await _request("GET", f"/API/v2/items/{quote(item_id)}/")


async def get_metadata(item_id: str) -> dict:
    return await _request("GET", f"/API/v2/items/{quote(item_id)}/metadata/")


async def get_formats(item_id: str) -> dict:
    """Shapes on an item, each with a download_uri and its backing files."""
    return await _request("GET", f"/API/v2/items/{quote(item_id)}/formats/")


async def resolve_download_url(item_id: str, shape_id: Optional[str] = None) -> Optional[str]:
    """
    Turn an item (optionally a specific shape) into a directly-fetchable URL.

    The Portal answers /vs/item/download/ with a 302 to a presigned S3 URL good
    for one hour. We read the Location rather than following it, so the caller
    can hand that URL to something else -- a downloader, or Cantemo's own
    import-by-uri -- without re-authenticating against the Portal.

    Passing no shape_id uses the item's "original" shape when there is one.
    """
    base = _require()
    if shape_id is None:
        formats = await get_formats(item_id)
        shapes = (formats or {}).get("formats") or []
        if not shapes:
            _logger.error("[cantemo] item %s has no shapes to download", item_id)
            return None
        original = next((s for s in shapes if s.get("name") == "original"), shapes[0])
        shape_id = original.get("id")

    url = f"{base}/vs/item/download/{quote(item_id)}/?shape={quote(str(shape_id))}"
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
        resp = await client.get(url, headers=_headers())
        if resp.status_code in (301, 302, 303, 307, 308):
            return resp.headers.get("location")
        if resp.is_success:
            return url  # served inline rather than redirected
    _logger.error("[cantemo] download for %s -> %s", item_id, resp.status_code)
    return None


async def download_to(item_id: str, dest_path: str, shape_id: Optional[str] = None) -> Optional[str]:
    """Fetch an item's media to a local path. Returns the path, or None."""
    src = await resolve_download_url(item_id, shape_id)
    if not src:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        async with client.stream("GET", src) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 20):
                    fh.write(chunk)
    return dest_path


# --- write ----------------------------------------------------------------

async def create_placeholder(
    title: str,
    group_name: Optional[str] = None,
    fields: Optional[list[dict]] = None,
    groups: Optional[list[dict]] = None,
) -> dict:
    """
    Create an empty item to import media into.

    `fields` are [{"name": ..., "value": ...}]; field names are Portal's
    internal ids (portal_mfNNNNNN) except for built-ins such as "title".
    """
    metadata: dict = {"fields": [{"name": "title", "value": title}] + list(fields or [])}
    if group_name:
        metadata["group_name"] = group_name
    if groups:
        metadata["groups"] = groups
    return await _request("POST", "/API/v2/items/", json={"metadata": metadata})


async def import_uri(
    item_id: str,
    uri: str,
    notranscode: bool = False,
    ingestprofiles: Optional[str] = None,
    tags: Optional[str] = None,
) -> dict:
    """
    Attach media to an existing item by URI -- the Portal copies it in.

    This is how a Conductor output lands in the MAM: get_job_outputs hands back
    signed URLs, and those go straight in here without staging the bytes through
    us. Note the API's own default for notranscode is true; we default it false
    so ingested media gets its proxies like anything else.
    """
    params: dict = {"uri": uri, "notranscode": str(bool(notranscode)).lower()}
    if ingestprofiles:
        params["ingestprofiles"] = ingestprofiles
    if tags:
        params["tags"] = tags
    return await _request("POST", f"/API/v2/items/{quote(item_id)}/import/", params=params)


async def set_metadata(item_id: str, fields: list[dict], group_name: Optional[str] = None) -> Any:
    metadata: dict = {"fields": fields}
    if group_name:
        metadata["group_name"] = group_name
    return await _request(
        "PUT", f"/API/v2/items/{quote(item_id)}/metadata/", json={"metadata": metadata}
    )


async def batch_set_metadata(
    item_ids: list[str],
    fields: list[dict],
    group_name: Optional[str] = None,
) -> Any:
    """
    Update many items in one call. Applied in the background after this returns,
    so a read-back immediately afterwards may still show the old values.
    """
    metadata: dict = {"fields": fields}
    if group_name:
        metadata["group_name"] = group_name
    return await _request(
        "PUT", "/API/v2/items/batch/metadata/", json={"item_ids": item_ids, "metadata": metadata}
    )


async def create_relation(
    parent_id: str,
    child_id: str,
    relation_type: str = "unknown",
    metadata: Optional[dict] = None,
) -> Any:
    """
    Link two items. This is how provenance is recorded as real graph edges
    rather than text stuffed in a description -- a trained LoRA relates back to
    every source asset, and a generated image relates back to the LoRA.
    """
    params = {"type": relation_type}
    body = {"metadata": metadata} if metadata else None
    return await _request(
        "POST",
        f"/API/v2/items/{quote(parent_id)}/relation/{quote(child_id)}/",
        params=params,
        json=body,
    )


async def get_relations(item_id: str) -> Any:
    return await _request("GET", f"/API/v2/items/{quote(item_id)}/relation/")


# --- metadata schema ------------------------------------------------------

async def list_metadata_groups(limit: int = 200) -> Any:
    return await _request("GET", f"/API/v2/metadata-schema/groups/?limit={int(limit)}")


async def get_metadata_group(name: str) -> Any:
    return await _request("GET", f"/API/v2/metadata-schema/groups/{quote(name)}/")


async def list_metadata_fields(limit: int = 500) -> Any:
    return await _request("GET", f"/API/v2/metadata-schema/fields/?limit={int(limit)}")


async def field_types() -> Any:
    return await _request("GET", "/API/v2/metadata-schema/fieldtypes/")
