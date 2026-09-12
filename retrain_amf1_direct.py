"""
One-off: retrain the Aston Martin F1 LoRA ("amf1") through Conductor for AI's
own training endpoint, using the 2026-09-10 contract change (dataset creation
removed -- submit directly with dataset_ids=[] and an `inputs` map of
asset_id -> caption).

Source images are VX-4427's original 8 training assets in Cantemo (read via
VX-4427's `lora_trained_from` relation edges), already downloaded locally.
Captions were written by hand (Claude, viewing each image) rather than
generated at request time -- this script is deliberately NOT wired into
Samsyn; Mark asked for a fast one-off retrain, not new automation.

Usage:
    python retrain_amf1_direct.py            # dry run: uploads + prints the
                                              # body it WOULD submit
    python retrain_amf1_direct.py --go       # actually submits (spends GPU time)
"""

import argparse
import asyncio
import json
import os

import conductor_ai_client as ai
import conductor_ai_pipeline as pipeline
from conductor_ai_pipeline import _upload_one_file

FLUX_SCHNELL_T5FP8 = "019a7ed4-df56-7b51-a0bc-5e5e063a42ac"  # same model Gotham_City trained against

TRAIN_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "f1_train")

CAPTIONS = {
    "VX-4408.jpg": "amf1 race car, rear three-quarter view, teal and black Aston Martin "
                   "Formula 1 car, aramco and cognizant sponsor livery, Pirelli tires with "
                   "yellow rim detailing, white studio background, product photography",
    "VX-4409.jpg": "amf1 race car, full side profile, teal and black Aston Martin Formula 1 "
                   "car, aramco cognizant peroni livery, white studio background, clean "
                   "automotive photography",
    "VX-4410.jpg": "amf1 race car, front head-on view, teal Aston Martin Formula 1 car, "
                   "aramco sponsor branding on front wing, black cockpit halo, white studio "
                   "background",
    "VX-4411.jpg": "amf1 race car, elevated front three-quarter view, teal Aston Martin "
                   "Formula 1 car, car number 18, aramco cognizant peroni sponsor livery, "
                   "soft grey studio background",
    "VX-4412.jpg": "amf1 race car, side profile view, teal and black Aston Martin Formula 1 "
                   "car, aramco cognizant crypto.com sponsor livery, white studio background",
    "VX-4413.jpg": "amf1 race car, die-cast scale model, teal Aston Martin Formula 1 car, "
                   "car number 14, aramco cognizant valvoline sponsor livery, glossy finish, "
                   "white background, product photography",
    "VX-4414.jpg": "amf1 race car, high-angle overhead three-quarter view, dark teal Aston "
                   "Martin Formula 1 car, car number 14, aramco cognizant valvoline sponsor "
                   "livery, detailed suspension and bodywork, white studio background, high "
                   "resolution",
    "VX-4416.jpg": "amf1 race car, symmetric top-down front view, dark teal Aston Martin "
                   "Formula 1 car, car number 14, aramco cognizant xerox netapp sponsor "
                   "livery, dramatic teal gradient background, studio photography",
}


async def main(go: bool) -> None:
    pid = await pipeline.project_id()
    print(f"project_id: {pid}")

    asset_ids: dict[str, str] = {}
    for fname in CAPTIONS:
        path = os.path.join(TRAIN_DIR, fname)
        asset_id = await _upload_one_file(pid, path)
        asset_ids[asset_id] = CAPTIONS[fname]
        print(f"uploaded {fname} -> asset {asset_id}")

    settings = {
        # Job 00005 (name with spaces, "Aston Martin F1 Livery v1") sat at
        # "created" indefinitely. Mark found the dashboard UI silently rejects
        # Submit on a spaced name -- retrying with underscores to see if the
        # API path has the same undocumented constraint.
        "lora_name": "Aston_Martin_F1_Livery_v1",
        "trigger_word": "amf1",
        "num_epochs": 8,
        "batch_size": 1,
        "network_dimension": 32,
        "network_alpha": 16,
        "num_repeats": 10,
        "learning_rate": "1e-4",
        "seed": 42,
        "sample_every_n_epochs": 1,
        "max_runtime_minutes": 60,
    }

    body_preview = {
        "dataset_ids": [],
        "model_id": FLUX_SCHNELL_T5FP8,
        "settings": settings,
        "inputs": asset_ids,
    }
    print(json.dumps(body_preview, indent=2))

    if not go:
        print("\nDry run only -- pass --go to actually submit the training job.")
        return

    resp = await ai.submit_training(pid, [], FLUX_SCHNELL_T5FP8, settings, inputs=asset_ids)
    print("\nSUBMITTED:")
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="Actually submit (spends GPU time).")
    args = parser.parse_args()
    asyncio.run(main(args.go))
