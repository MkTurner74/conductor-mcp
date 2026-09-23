"""
One-off: validate Conductor for AI's two fixes (Jesse Lehrman, 2026-09-23)
before their week-long deployment freeze for Fully Connected.

  Fix 1: a broken job no longer blocks other jobs.
  Fix 2: `machine` is validated at submission; a bad GPU spec returns 400.

Replays the jobs that broke the system (00004/00005: spaced lora_name, sat at
`created` forever) alongside a clean control job, and checks the control job
actually starts moving. Reuses the amf1 training assets already uploaded for
job 00006 (read back from that job's `inputs`), so nothing is re-uploaded.

Usage:
    python validate_ai_fixes.py            # dry run: prints every body it WOULD submit
    python validate_ai_fixes.py --go       # submits (2 x 1-epoch jobs, well under $1)

Needs CONDUCTOR_AI_API_KEY in the environment (the 09-08 key expired 09-15).
"""

import argparse
import asyncio
import json
import time

import conductor_ai_client as ai
import conductor_ai_pipeline as pipeline
from conductor_ai_client import ConductorAIError

FLUX_SCHNELL_T5FP8 = "019a7ed4-df56-7b51-a0bc-5e5e063a42ac"
SOURCE_JOB_NAME = "Aston_Martin_f1_via_UX"  # job 00006, the one that succeeded

VALID_MACHINE = {"gpu": {"type": "L40", "count": 1},
                 "cpu": {"type": "some_cpu", "count": 16}, "memory": "64Gi"}
BAD_MACHINE = {"gpu": {"type": "NOT_A_REAL_GPU", "count": 1},
               "cpu": {"type": "some_cpu", "count": 16}, "memory": "64Gi"}

POLL_MINUTES = 15
STUCK_AFTER_MINUTES = 10  # rule of thumb from the 09-10/14 sessions


def _settings(name: str) -> dict:
    return {
        "lora_name": name, "trigger_word": "amf1", "num_epochs": 1, "batch_size": 1,
        "network_dimension": 32, "network_alpha": 16, "num_repeats": 10,
        "learning_rate": "1e-4", "seed": 42, "sample_every_n_epochs": 1,
        "max_runtime_minutes": 20,
    }


async def _submit(pid: str, settings: dict, inputs: dict, machine: dict) -> dict:
    """Raw POST so `machine` can ride along without changing the client's signature."""
    body = {"dataset_ids": [], "model_id": FLUX_SCHNELL_T5FP8, "settings": settings,
            "inputs": inputs, "machine": machine}
    return await ai._request("POST", f"/core/v1/projects/{pid}/lora-trainings", json_body=body)


def _rows(resp) -> list:
    if isinstance(resp, list):
        return resp
    return (resp or {}).get("data") or (resp or {}).get("lora_trainings") or []


def _brief(row: dict) -> str:
    s = row.get("settings") or {}
    return (f"{row.get('id')}  status={row.get('status')!s:<10} "
            f"name={s.get('lora_name') or row.get('name')!r}  created={row.get('created_at')}")


async def main(go: bool) -> None:
    pid = await pipeline.project_id()
    print(f"project_id: {pid}\n")

    # 1. Where do the old stuck jobs stand now?
    trainings = _rows(await ai.list_trainings(pid, limit=50))
    print("== Existing training jobs (look for 00004/00005/00007/00008 -- still `created`?)")
    for row in trainings[:15]:
        print("  " + _brief(row))

    source = next((r for r in trainings
                   if (r.get("settings") or {}).get("lora_name") == SOURCE_JOB_NAME
                   and str(r.get("status")).lower() in ("completed", "succeeded", "success")), None)
    if not source:
        raise SystemExit(f"No completed job named {SOURCE_JOB_NAME} to borrow inputs from.")
    full = await ai.get_training(pid, str(source["id"]))
    inputs = full.get("inputs") or {}
    if not inputs:
        raise SystemExit("Job 00006 doesn't echo its `inputs` back -- re-upload needed "
                         "(see retrain_amf1_direct.py). Stopping.")
    print(f"\nReusing {len(inputs)} assets from job {source['id']}\n")

    plan = [
        ("A  bad GPU spec -> expect 400", _settings("validate_bad_gpu"), BAD_MACHINE),
        ("B  replay of 00004/00005 (spaced name)", _settings("Aston Martin F1 Livery v1"), VALID_MACHINE),
        ("C  clean control -> must start moving", _settings("validate_control_l40"), VALID_MACHINE),
    ]
    if not go:
        for label, settings, machine in plan:
            print(f"== {label}")
            print(json.dumps({"settings": settings, "machine": machine,
                              "inputs": f"<{len(inputs)} assets>"}, indent=2))
        print("\nDry run only -- pass --go to submit.")
        return

    # 2. Submit all three.
    submitted: dict[str, str] = {}
    for label, settings, machine in plan:
        try:
            resp = await _submit(pid, settings, inputs, machine)
            job_id = str((resp or {}).get("id") or "")
            submitted[label] = job_id
            verdict = "FAIL (accepted a bad GPU)" if label.startswith("A") else "accepted"
            print(f"{label}: {verdict}  id={job_id}")
        except ConductorAIError as e:
            ok = label.startswith("A") and e.status == 400
            print(f"{label}: {'PASS' if ok else 'UNEXPECTED'} -- HTTP {e.status}: {e}")

    # 3. Poll B and C. Fix 1 passes if C leaves `created` within ~10 min
    #    regardless of what B does.
    started = time.time()
    watch = {k: v for k, v in submitted.items() if not k.startswith("A") and v}
    while watch and time.time() - started < POLL_MINUTES * 60:
        await asyncio.sleep(30)
        mins = (time.time() - started) / 60
        line = [f"t+{mins:4.1f}m"]
        for label, job_id in list(watch.items()):
            row = await ai.get_training(pid, job_id)
            ep = await ai.list_epochs(pid, job_id, limit=1, order_by_asc=False) or {}
            epochs = ep.get("epochs") or []
            line.append(f"{label[:1]}={row.get('status')} epochs={len(epochs)}")
        print("  ".join(line))

    print("\nRead-out for Jesse:")
    print("  Fix 2 = PASS if A returned HTTP 400.")
    print(f"  Fix 1 = PASS if C left `created` (running/epoch progress) within {STUCK_AFTER_MINUTES} min,")
    print("          whatever B did. B either dispatching or failing cleanly is a bonus.")
    print("  Cancel anything still `created` afterwards (the cancel endpoint used to 500 -- worth noting).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="Actually submit (spends GPU time).")
    args = parser.parse_args()
    asyncio.run(main(args.go))
