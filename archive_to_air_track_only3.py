"""
Third fix: prov_source_assets has a 64-char cap in Cantemo's schema — 26
comma-joined item ids blew past it (the F1 demo only ever had 8). Write a
short summary string there instead, and rely on the per-source create_relation
edges (looped separately, unaffected by that field's length) for full
provenance. Job 00028 is still training (confirmed running, 0 failed) — just
takes longer than SDXL/8-epoch jobs this session has seen before (Coda's
audio jobs, or the F1 LoRA's own timing was never itself logged). Longer
poll window this time; still does not resubmit.
"""
import asyncio
import json
import sys
import time

import httpx

import cantemo_client as cantemo
import conductor_client as conductor
import lora_pipeline as lp

JOB_ID = "00028"
ITEM_ID = "VX-4494"
ITEM_IDS = [f"VX-{n}" for n in range(4468, 4494)]
LABEL = "Tears of Steel - House Style v1"
TRIGGER_WORD = "tos_style"
BASE_MODEL = "sdxl1-kohya"
SOURCE_SUMMARY = f"{len(ITEM_IDS)} keyframes ({ITEM_IDS[0]}..{ITEM_IDS[-1]})"  # 32 chars, under the 64-char cap


async def try_track() -> bool:
    try:
        tracked_fields, _ = await lp._writable_provenance_fields(
            {
                "provenance_kind": "lora",
                "status": lp.STATUS_SUBMITTED,
                "label": LABEL,
                "base_model": BASE_MODEL,
                "trigger_word": TRIGGER_WORD,
                "job_id": JOB_ID,
                "source_asset_ids": SOURCE_SUMMARY,
            }
        )
        await cantemo.set_metadata(ITEM_ID, tracked_fields, group_name=lp.PROVENANCE_GROUP)
        print("   metadata write OK")
    except httpx.HTTPStatusError as exc:
        print(f"   set_metadata still failing: {exc.response.text}", file=sys.stderr)
        return False

    linked, failed = [], []
    for src in ITEM_IDS:
        try:
            await cantemo.create_relation(ITEM_ID, src, relation_type=lp.REL_TRAINED_FROM)
            linked.append(src)
        except Exception as exc:
            failed.append({"item_id": src, "error": str(exc)})
    print(f"   relations linked={len(linked)} failed={len(failed)}")
    if failed:
        print(f"   relation failures: {json.dumps(failed[:5], default=str)}")
    return True


async def main():
    print(f"== 6. Tracking job {JOB_ID} onto {ITEM_ID} (fixed field length) ==")
    tracked_ok = await try_track()

    print("== 7. Polling job status (longer window: 55 min) ==")
    deadline = time.time() + 55 * 60
    terminal_status = None
    while time.time() < deadline:
        status = await lp.job_status(JOB_ID)
        raw = str(status.get("status") or "").lower()
        print(f"   [{time.strftime('%H:%M:%S')}] status={raw} terminal={status.get('terminal')} progress={status.get('progress')}")
        if tracked_ok:
            try:
                await lp.sync_status_to_mam(ITEM_ID, JOB_ID)
            except Exception as exc:
                print(f"   sync_status_to_mam failed: {exc}", file=sys.stderr)
        if status.get("terminal"):
            terminal_status = raw
            break
        await asyncio.sleep(25)

    print(f"== Final status: {terminal_status} ==")
    if terminal_status in ("success", "complete", "completed"):
        if tracked_ok:
            print("== 8. Finalizing (attaching weights) ==")
            try:
                fin = await lp.finalize_tracked_lora(ITEM_ID, JOB_ID)
                print(f"   {fin}")
            except httpx.HTTPStatusError as exc:
                print(f"   finalize error body: {exc.response.text}", file=sys.stderr)
            print("== 9. Stamping source assets ==")
            stamp = await lp.stamp_source_assets(
                source_item_ids=ITEM_IDS, label=LABEL, job_id=JOB_ID,
                base_model=BASE_MODEL, trigger_word=TRIGGER_WORD,
            )
            print(f"   {stamp}")
        else:
            print("   Tracking never succeeded — fetching raw job outputs instead.")
            outputs = await conductor.get_job_outputs(JOB_ID)
            print(json.dumps(outputs, indent=2, default=str)[:3000])
        print(f"\nDONE. item_id = {ITEM_ID} job_id = {JOB_ID}")
    elif terminal_status:
        print(f"\nJob reached a terminal but non-success state: {terminal_status}. item_id={ITEM_ID} job_id={JOB_ID}")
        try:
            outputs = await conductor.get_job_outputs(JOB_ID)
            detail = await lp.training_failure_detail(outputs)
            print(f"failure detail: {detail}")
        except Exception as exc:
            print(f"could not fetch failure detail: {exc}")
    else:
        print(f"\nStill not terminal after 55 more minutes — genuinely long-running or stuck. job_id={JOB_ID}")


if __name__ == "__main__":
    asyncio.run(main())
