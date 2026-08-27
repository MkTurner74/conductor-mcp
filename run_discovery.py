"""
One-shot Conductor discovery run.

Answers the two things the LoRA pipeline is currently guessing at:

  1. What is actually inside the kohya package -- the entry point names, the
     python location, the sd-scripts layout. TRAIN_COMMAND_TEMPLATE and
     INFER_COMMAND_TEMPLATE in lora_pipeline.py are built from expectation, not
     observation, until this runs.
  2. Where an uploaded file lands on the render node. A probe file is uploaded
     with the job purely so the listing can find it, which is what node_path()
     currently assumes rather than knows.

Runs on the cheapest CPU instance with no GPU and no model load, so it costs
pennies. It still submits a real job and spends real money -- that is why this
is a script you run deliberately rather than something the pipeline does for
you.

Usage (PowerShell):

    & "C:\\Users\\mktur\\code\\conductor-mcp\\.venv\\Scripts\\python.exe" `
      "C:\\Users\\mktur\\code\\conductor-mcp\\run_discovery.py"

Add --dry-run to build the job and print it without submitting.
Add --job <jid> to skip submission and just fetch results for an existing job.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Quieten httpx's per-request logging -- it echoes the Conductor private key,
# which is sent as a URL query parameter by the OAuth flow in conductor_client.
logging.getLogger("httpx").setLevel(logging.WARNING)

os.environ.setdefault("CONDUCTOR_API_KEY_FILE", os.path.join(REPO, "conductor_key.json"))

import conductor_client as conductor  # noqa: E402
import conductor_render  # noqa: E402
import lora_pipeline as lp  # noqa: E402

POLL_SECONDS = 20
MAX_WAIT_SECONDS = 20 * 60
DASHBOARD = os.getenv("CONDUCTOR_DASHBOARD_URL", "https://dashboard.conductortech.com")

TERMINAL = {"success", "failed", "killed", "complete", "completed"}


def make_probe() -> str:
    """A small uploaded file whose location on the node reveals the path mapping."""
    workdir = os.path.join(tempfile.gettempdir(), "conductor-lora-discovery")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "DISCOVERY_PROBE.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"conductor upload path probe {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return path


async def wait_for(jid: str) -> dict:
    """Poll until the job reaches a terminal state, or we give up waiting."""
    started = time.time()
    last = None
    while time.time() - started < MAX_WAIT_SECONDS:
        status = await conductor_render.render_status(jid)
        state = str(status.get("status") or "").lower()
        if state != last:
            elapsed = int(time.time() - started)
            print(f"  [{elapsed:>4}s] status: {status.get('status')} "
                  f"(running={status.get('running')} pending={status.get('pending')} "
                  f"success={status.get('success')} failed={status.get('failed')})")
            last = state
        if state in TERMINAL:
            return status
        if state == "not_found":
            # Conductor can lag briefly between submit and the job appearing.
            pass
        await asyncio.sleep(POLL_SECONDS)
    print(f"  gave up waiting after {MAX_WAIT_SECONDS}s -- the job may still be running")
    return await conductor_render.render_status(jid)


async def job_ids(jid: str) -> dict:
    """
    Resolve a padded jid to Conductor's internal numeric ids.

    The status call returns neither the task id nor anything that links to the
    log, but the raw job record carries `id` and `task_keys`, and
    /api/v1/tasks/{key} resolves from there. Used only to print useful links.
    """
    try:
        jobs = await conductor.list_jobs()
        data = jobs.get("data", jobs) if isinstance(jobs, dict) else jobs
        job = next((j for j in data if str(j.get("jid")) == str(jid).zfill(5)), None)
        if not job:
            return {}
        keys = job.get("task_keys") or []
        return {"job_id": job.get("id"), "task_keys": keys, "output_path": job.get("output_path")}
    except Exception as exc:
        _log_err(f"could not resolve internal ids: {exc}")
        return {}


def _log_err(msg: str) -> None:
    print(f"  ! {msg}")


async def fetch_log(jid: str) -> str:
    """
    Try to pull the task log through the API.

    Confirmed 2026-08-27 against job 00005: this does not currently work on this
    account. /get_log_file answers 500 for every parameter shape tried --
    task id "000" (the real tid, per /api/v1/tasks/{key}), the padded and
    unpadded jid, and the internal numeric job/task ids. No log field appears on
    the task or execution record either. The dashboard UI shows the log fine, so
    read it there rather than trusting this path.
    """
    for task_id in ("000", "001", "1", "01", "0"):
        try:
            data = await conductor.get_task_log(jid, task_id)
        except Exception:
            continue
        text = data if isinstance(data, str) else json.dumps(data)
        if text and text.strip() not in ("", "{}", "null", '""') and not text.lstrip().startswith("<!DOCTYPE"):
            print(f"  (log retrieved as task {task_id})")
            return text
    return ""


def report(log_text: str) -> None:
    """Print the sections the discovery command emitted, and what they settle."""
    if not log_text:
        print("\nNo log came back through the API -- expected, see fetch_log()'s note.")
        print("Read it in the dashboard instead:")
        print(f"  {DASHBOARD}/jobs   -> open the job -> the task -> Log")
        print("\nCopy everything between ===ENV=== and ===DONE=== back to Claude.")
        return

    print("\n" + "=" * 72)
    print("DISCOVERY OUTPUT")
    print("=" * 72)
    print(log_text)

    print("\n" + "=" * 72)
    print("WHAT TO DO WITH THIS")
    print("=" * 72)
    print("""
  ===KOHYA_BIN=== / ===TRAIN_SCRIPTS===
      The real entry point names. If there is no bare `accelerate` or
      `sdxl_train_network.py` in $KOHYA_PATH, update TRAIN_COMMAND_TEMPLATE and
      INFER_COMMAND_TEMPLATE in lora_pipeline.py to whatever is actually there.

  ===UPLOAD_LOCATION===
      Where DISCOVERY_PROBE.txt landed. Compare it to what node_path() would
      have produced for the local path. If they differ, fix node_path() -- a
      mismatch fails on the node, not at submission, so it is silent until a
      real training run burns GPU minutes.

  ===MODEL===
      Confirms the SDXL base model really is on the node's disk, which is the
      whole reason the no-egress constraint stopped mattering.
""")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Conductor kohya discovery job")
    parser.add_argument("--dry-run", action="store_true", help="build and print the job, submit nothing")
    parser.add_argument("--job", help="skip submission; fetch results for this existing job id")
    parser.add_argument("--probe-env", action="store_true",
                        help="run the env probe instead: source pkg_env.sh and report what lands on PATH")
    args = parser.parse_args()

    if args.job:
        jid = args.job
        print(f"Fetching results for existing job {jid}")
    elif args.probe_env:
        print("Building env probe job (sources $ENV_FILE on the node)...")
        result = await lp.submit_env_probe(dry_run=args.dry_run)
        if args.dry_run:
            print("\nDRY RUN -- nothing submitted.\n")
            print(result["job_args"]["tasks_data"][0]["command"])
            return 0
        jid = result.get("jid")
        if not jid:
            print(json.dumps(result, indent=2, default=str))
            return 1
        print(f"\nSubmitted. Job {jid}\n")
    else:
        probe = make_probe()
        print(f"Probe file: {probe}")
        print("Building discovery job...")
        result = await lp.submit_discovery(probe, dry_run=args.dry_run)

        if args.dry_run:
            ja = result["job_args"]
            print("\nDRY RUN -- nothing submitted.\n")
            for key in ("job_title", "project", "instance_type", "software_package_ids",
                        "preemptible", "output_path", "upload_paths"):
                print(f"  {key}: {ja[key]}")
            print(f"\n  command:\n    {ja['tasks_data'][0]['command']}")
            return 0

        jid = result.get("jid")
        if not jid:
            print("Submission returned no job id:")
            print(json.dumps(result, indent=2, default=str))
            return 1
        print(f"\nSubmitted. Job {jid}")
        print(f"  {DASHBOARD}/jobs\n")

    print("Waiting for the job to finish...")
    status = await wait_for(str(jid))
    print(f"\nFinal status: {status.get('status')}  ({status.get('status_description') or 'no description'})")

    ids = await job_ids(str(jid))
    if ids:
        print(f"  internal job id: {ids.get('job_id')}  task_keys: {ids.get('task_keys')}")

    print("\nFetching task log...")
    report(await fetch_log(str(jid)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted. The job may still be running -- check the dashboard.")
        sys.exit(130)
