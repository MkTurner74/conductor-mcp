"""
Two remaining well-specified Archive to Air tasks:
1. Ingest the Tears of Steel source video into Cantemo + write real MDDF/EIDR-
   style metadata (using the actual field names/limits confirmed from the
   live ArchiveToAir group — note Release_year is capitalized).
2. Actually run the demo "generate ident" workflow for real: inference against
   the trained LoRA (VX-4494), land the result, file it into Deliverables.
"""
import asyncio
import json
import sys
import time

import boto3

import cantemo_client as cantemo
import conductor_client as conductor
import lora_pipeline as lp

S3_BUCKET = "botverse-test-fixtures"
VIDEO_PATH = (
    "C:/Users/mktur/OneDrive - Entertainment Technologists Inc/Ai Companion Docs/"
    "projects/skywalker-coda-samsyn/sample-media/content/tears_of_steel_720p.mov"
)
VIDEO_S3_KEY = "archive-to-air/tears_of_steel_720p.mov"
ARCHIVE_ROOT_COLLECTION = "VX-2610"
DELIVERABLES_COLLECTION = "VX-2612"
LORA_ITEM_ID = "VX-4494"


def upload_video_and_presign() -> str:
    session = boto3.Session(profile_name="botverse")
    s3 = session.client("s3")
    s3.upload_file(VIDEO_PATH, S3_BUCKET, VIDEO_S3_KEY)
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": VIDEO_S3_KEY}, ExpiresIn=3600
    )


async def ingest_source_video() -> str:
    print("== A1. Uploading Tears of Steel video to S3 ==")
    url = upload_video_and_presign()
    print("   uploaded, presigned URL ready")

    print("== A2. Creating the Cantemo item ==")
    item = await cantemo.create_placeholder(title="Tears of Steel (archive master)")
    item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
    if not item_id:
        raise RuntimeError(f"Could not read new item id: {item}")
    print(f"   item_id = {item_id}")

    print("== A3. Importing the video file (this may take a bit — 355MB) ==")
    await cantemo.import_uri(item_id, url, notranscode=False)
    print("   import_uri call accepted")

    print("== A4. Filing into the Archive to Air collection ==")
    res = await cantemo.add_to_collection(ARCHIVE_ROOT_COLLECTION, [item_id])
    print(f"   {res}")

    print("== A5. Writing MDDF/EIDR-style metadata (real field names, note Release_year capitalized) ==")
    fields = [
        {"name": "title_eidr", "value": "10.5240/7A3F-9E21-C4B8-01D6-55F2-Q"},  # fake, correctly formatted, not a real registered EIDR
        {"name": "edit_eidr", "value": "10.5240/2B6C-4D8E-A1F0-33C9-77E4-X"},   # fake, distinct edit-level id
        {"name": "work_type", "value": "short_film"},
        {"name": "Release_year", "value": "2012"},   # real — Tears of Steel's actual release year
        {"name": "territory_air", "value": "Worldwide"},
        {"name": "rating_system", "value": "Not Rated"},
        {"name": "rating_value", "value": "Not Rated"},
        {"name": "alt_identifier", "value": "Blender Foundation / Project Mango"},  # real
    ]
    await cantemo.set_metadata(item_id, fields, group_name="ArchiveToAir")
    print("   metadata write OK")
    return item_id


async def run_demo_inference() -> dict:
    print("\n== B1. Running the demo inference workflow for real ==")
    workdir = "C:/Users/mktur/AppData/Local/Temp/archive-to-air-infer"
    prompt = (
        "tos_style wide title-card establishing shot, Amsterdam skyline at dusk, "
        "sci-fi brand ident card, cinematic blue-grey grade"
    )
    real = await lp.submit_inference(
        dry_run=False, lora_item_id=LORA_ITEM_ID, prompt=prompt, workdir=workdir,
        model="sdxl", count=4, steps=30, seed=42, strength=1.25,
    )
    print(json.dumps(real, indent=2, default=str)[:2000])
    job_id = real.get("jid") or real.get("job_id")
    print(f"   job_id = {job_id}")
    if not job_id:
        raise RuntimeError("No job id returned from inference submission")

    print("== B2. Polling (inference is short) ==")
    deadline = time.time() + 20 * 60
    terminal_status = None
    while time.time() < deadline:
        status = await lp.job_status(str(job_id))
        raw = str(status.get("status") or "").lower()
        print(f"   [{time.strftime('%H:%M:%S')}] status={raw} terminal={status.get('terminal')}")
        if status.get("terminal"):
            terminal_status = raw
            break
        await asyncio.sleep(15)

    print(f"== Final status: {terminal_status} ==")
    if terminal_status not in ("success", "complete", "completed"):
        outputs = await conductor.get_job_outputs(str(job_id))
        detail = await lp.training_failure_detail(outputs)
        return {"ok": False, "job_id": job_id, "status": terminal_status, "detail": detail}

    print("== B3. Landing the generated ident image(s) in the MAM + filing into Deliverables ==")
    landed = await lp.ingest_generated_images(
        job_id=str(job_id), prompt=prompt, lora_item_id=LORA_ITEM_ID,
        base_model="sdxl1-kohya", created_by="MkT", collection="Deliverables",
    )
    print(f"   {json.dumps(landed, default=str)[:1500]}")
    return {"ok": landed.get("ok", False), "job_id": job_id, "items": landed.get("items", [])}


async def main():
    try:
        video_item_id = await ingest_source_video()
        print(f"\nVIDEO ITEM: {video_item_id}\n")
    except Exception as exc:
        print(f"Video ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        video_item_id = None

    result = await run_demo_inference()
    print(f"\nDEMO RUN RESULT: {json.dumps(result, default=str)}")
    print(f"VIDEO ITEM: {video_item_id}")


if __name__ == "__main__":
    asyncio.run(main())
