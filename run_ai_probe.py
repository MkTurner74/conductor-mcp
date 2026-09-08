"""
Conductor for AI — first-contact probe.

Run this the moment the API key arrives. It answers, in one pass, every
question the OpenAPI document could not:

  1. which Authorization header style the service actually accepts
  2. what projects exist, and which one we are scoped to
  3. whether base models are pre-registered (SDXL / Flux) or need registering
  4. what LoRAs, datasets and assets the project already holds
  5. (optional) the real request body for dataset creation, by asking and
     reading the server's own complaint
  6. (optional, SPENDS MONEY) a live generation end to end, recording the
     status vocabulary, the timings and the cost

Everything except steps 5 and 6 is read-only and free.

    # read-only
    python run_ai_probe.py

    # also probe the undocumented dataset body (creates an empty dataset)
    python run_ai_probe.py --dataset-probe

    # also run one real generation (GPU time, real money)
    python run_ai_probe.py --generate "an Aston Martin F1 car on a wet track"

A markdown report is written next to the repo (--report to change the path) so
the findings survive the terminal session — and so they can be pasted straight
into the project doc.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

import conductor_ai_client as ai
import conductor_ai_pipeline as pipeline

REPORT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conductor-ai-probe-report.md")

# Windows consoles default to cp1252, which cannot print an arrow or an em
# dash — and an encoding error must never be what stops a probe mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def block(label: str, payload) -> None:
    say(f"\n### {label}\n")
    say("```json")
    say(json.dumps(payload, indent=2, default=str)[:4000])
    say("```")


async def probe_auth() -> bool:
    say("## 1. Auth")
    if not ai.configured():
        say("\n**CONDUCTOR_AI_API_KEY is not set.** Nothing else can run.")
        say("\nSet it and re-run:\n")
        say("```")
        say('  PowerShell:  $env:CONDUCTOR_AI_API_KEY = "<key>"')
        say('  bash:        export CONDUCTOR_AI_API_KEY="<key>"')
        say("```")
        return False
    say(f"\n- host: `{ai.api_url()}`")
    say(f"- key: `…{ai.api_key()[-6:]}` ({len(ai.api_key())} chars)")
    try:
        projects = await ai.list_projects()
    except ai.ConductorAIError as exc:
        say(f"\n**FAILED** — {exc}")
        say("\nIf this is a 401 on BOTH header styles, the key is not valid for this API "
            "(the render service-account key is not — that was proven on 2026-09-03).")
        return False
    say(f"\n- accepted Authorization style: **{ai._auth_style}** "
        f"(`Authorization: {'<key>' if ai._auth_style == 'raw' else 'Bearer <key>'}`)")
    rows = projects.get("data") or []
    say(f"- projects visible: **{len(rows)}**")
    for r in rows:
        say(f"  - {r.get('name')} — `{r.get('id')}` (active={r.get('active')})")
    return True


async def probe_project(create: str = "") -> str:
    say("\n## 2. Project")
    if create:
        say(f"\nCreating project `{create}` …")
        made = await ai.create_project(create, description="Samsyn / ETI Conductor-for-AI work")
        block("POST /core/v1/projects", made)
        return str(made.get("id") or "")
    try:
        pid = await ai.resolve_project_id()
    except ai.ConductorAIError as exc:
        say(f"\n**Could not resolve a project** — {exc}")
        say("\nEither set CONDUCTOR_AI_PROJECT_ID / CONDUCTOR_AI_PROJECT, or re-run with "
            "`--create-project <name>`.")
        return ""
    say(f"\n- scoped to project `{pid}`")
    try:
        block("GET /core/v1/projects/{id}", await ai.get_project(pid))
    except ai.ConductorAIError as exc:
        say(f"(project detail read failed: {exc})")
    return pid


async def probe_models(pid: str) -> list[dict]:
    say("\n## 3. Base models")
    try:
        resp = await ai.list_base_models(pid)
    except ai.ConductorAIError as exc:
        say(f"\n**FAILED** — {exc}")
        return []
    rows = (resp or {}).get("data") or []
    say(f"\n- registered base models: **{len(rows)}**")
    if not rows:
        say("\n**None registered.** Either they are not pre-published for this account, or "
            "they need registering via `POST /base-models` (BaseModelPostRequest: name, "
            "model_name, model_family, model_path, engine_path, encoders_path, env_path, "
            "command…). Ask Conductor which is expected before guessing paths — those fields "
            "describe THEIR filesystem, not ours.")
    for r in rows:
        say(f"  - {r.get('name') or r.get('model_name')} — family `{r.get('model_family')}` — "
            f"`{r.get('id')}` published={r.get('publish')}")
    block("GET /core/v1/projects/{id}/base-models (first row, full shape)", rows[0] if rows else {})
    return rows


async def probe_loras(pid: str, models: list[dict]) -> None:
    say("\n## 4. LoRAs, datasets, assets")
    for m in models:
        mid = str(m.get("id") or "")
        if not mid:
            continue
        try:
            resp = await ai.list_loras(pid, mid)
            rows = (resp or {}).get("data") or []
            say(f"\n- `{m.get('name') or m.get('model_name')}` → **{len(rows)}** LoRA(s)")
            for r in rows[:10]:
                say(f"  - {r.get('name')} — trigger `{r.get('trigger_word')}` — `{r.get('id')}` "
                    f"(epoch {r.get('epoch')}/{r.get('total_epochs')})")
            if rows:
                block("lora-models row (full shape)", rows[0])
        except ai.ConductorAIError as exc:
            say(f"\n- `{mid}` LoRA list failed: {exc}")

    for label, call in (("datasets", ai.list_datasets), ("assets", ai.list_assets)):
        try:
            resp = await call(pid)
            rows = (resp or {}).get("data") or []
            say(f"\n- {label}: **{len(rows)}**")
            if rows:
                block(f"{label} row (full shape)", rows[0])
        except ai.ConductorAIError as exc:
            say(f"\n- {label} list failed: {exc}")


async def probe_dataset_body(pid: str) -> None:
    """The one write the spec cannot describe, asked directly of the server."""
    say("\n## 5. Dataset creation (undocumented request body)")
    name = f"samsyn-probe-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    say(f"\nPOST /assets/dataset with `{{\"name\": \"{name}\"}}` …")
    try:
        made = await ai.create_dataset(pid, name=name)
        block("201 response", made)
        say("\n**Body shape accepted.** `{name}` is enough to create; attaching assets is then "
            "`PUT /assets/dataset/{id}` with `asset_ids`.")
        ds_id = str((made or {}).get("id") or "")
        if ds_id:
            say(f"\nCreated dataset `{ds_id}` — delete it from the dashboard if it is in the way.")
    except ai.ConductorAIError as exc:
        say(f"\n**Rejected — and this is the useful part.** The server said:\n")
        say("```")
        say(str(exc.body)[:1500])
        say("```")
        say("\nAdjust `conductor_ai_client.create_dataset` to match, then re-run.")


async def probe_generation(pid: str, prompt: str, model: str, lora_id: str,
                           trigger_word: str, timeout_s: int) -> None:
    say("\n## 6. Live generation (this one costs money)")
    say(f"\n- prompt: `{prompt}`")
    started = time.time()
    sub = await pipeline.submit_generation(
        prompt=prompt, model=model, lora_id=lora_id, trigger_word=trigger_word,
        name="samsyn-probe",
    )
    block("POST /inferences", sub)
    if not sub.get("ok"):
        return
    inference_id = sub["inference_id"]

    seen: list[str] = []
    state = "running"
    status = {}
    while time.time() - started < timeout_s:
        status = await pipeline.generation_status(inference_id)
        raw = status.get("status") or ""
        if raw and (not seen or seen[-1] != raw):
            seen.append(raw)
            say(f"  · {int(time.time() - started):>4}s  {raw}")
        state = status.get("state") or "running"
        if state in ("success", "failed"):
            break
        await asyncio.sleep(5)

    elapsed = int(time.time() - started)
    say(f"\n- finished in **{elapsed}s** with state **{state}**")
    say(f"- status vocabulary observed: {seen}")
    say("  (fold any unfamiliar spelling into TERMINAL_OK / TERMINAL_BAD in conductor_ai_pipeline.py)")
    block("final status", status)

    try:
        outputs = await pipeline.generation_images(inference_id)
        say(f"\n- images returned: **{outputs['count']}**")
        block("outputs", outputs)
    except ai.ConductorAIError as exc:
        say(f"\n- output fetch failed: {exc}")

    say("\n**The number that matters:** wall-clock from submit to images. The kohya render-job "
        "path takes minutes of staging and cold start before a pixel exists; if this is "
        "materially faster, that is the whole reason to move.")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the Conductor for AI API")
    ap.add_argument("--create-project", default="", help="create a project with this name first")
    ap.add_argument("--dataset-probe", action="store_true", help="probe the dataset request body (writes)")
    ap.add_argument("--generate", default="", help="run ONE real generation with this prompt (spends money)")
    ap.add_argument("--model", default="", help="base model id/name/family for --generate")
    ap.add_argument("--lora-id", default="", help="optional LoRA id for --generate")
    ap.add_argument("--trigger-word", default="", help="the LoRA's trigger word")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for a generation")
    ap.add_argument("--report", default=REPORT_DEFAULT, help="where to write the markdown report")
    args = ap.parse_args()

    say(f"# Conductor for AI — probe report")
    say(f"\n*Run {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from {os.uname().nodename if hasattr(os, 'uname') else os.getenv('COMPUTERNAME', 'this machine')}*\n")

    # A failed probe is exactly the report worth keeping, so it is written on
    # every path out of here, not only the happy one.
    code = 0
    pid = ""
    try:
        if not await probe_auth():
            return 1
        pid = await probe_project(args.create_project)
        if not pid:
            return 1

        models = await probe_models(pid)
        await probe_loras(pid, models)

        if args.dataset_probe:
            await probe_dataset_body(pid)

        if args.generate:
            if not models:
                say("\n## 6. Live generation — SKIPPED: no base model to generate against.")
            else:
                await probe_generation(pid, args.generate, args.model, args.lora_id,
                                       args.trigger_word, args.timeout)

        say("\n## Next")
        say("\n- Put the working values into Railway env: `CONDUCTOR_AI_API_KEY`, "
            "`CONDUCTOR_AI_PROJECT_ID`" + (f" (=`{pid}`)" if pid else "") +
            (", `CONDUCTOR_AI_AUTH_STYLE`" if ai._auth_style else "") + ".")
        say("- Redeploy conductor-mcp (`railway up`) so Samsyn's palette picks up the new tools.")
        say("- Then run the Samsyn test workflow — see conductor-ai-api-runbook.md.")
    except Exception as exc:                      # noqa: BLE001 — the report is the point
        say(f"\n**Probe stopped on an unexpected error:** `{type(exc).__name__}: {exc}`")
        code = 1
    finally:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[report written to {args.report}]")
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
