"""
Cantemo MAM <-> Conductor LoRA round trip.

The CoreWeave IBC demo, end to end:

    select assets in Cantemo -> train a LoRA on CoreWeave GPUs via Conductor
    -> LoRA lands back in Cantemo with provenance -> select the LoRA, give a
    prompt -> generated images land back in Cantemo, still attributed.

Two findings shape this module, both worth knowing before changing it:

1. **Conductor ships kohya as a software package, with the base model baked in.**
   Products `kohya` (the host) plus `sdxl1-kohya` / `flux*-kohya` / `sd3*-kohya`
   (model plugins, vendor "griptape"). The plugin's environment carries
   MODEL_BASE_HOME pointing at an on-disk HuggingFace snapshot. That matters
   because CoreWeave render nodes have **no egress** -- nothing can be pulled
   from HuggingFace at run time. Only the training images need staging, via
   upload_paths. Nothing multi-gigabyte does.

2. **Submission goes through ciocore, not raw REST.** conductor_client.submit_job
   posts to /api/v1/jobs and carries a 404 caveat; the proven path is
   ciocore.conductor_submit.Submit, as used by conductor_render.py for the
   Houdini job that ran successfully on 2026-07-28.

Everything that spends GPU money defaults to dry_run=True. Callers opt in.

**Settled by discovery job 00005 (2026-08-27):**
  * $KOHYA_PATH is the sd-scripts repo root. sdxl_train_network.py and
    sdxl_gen_img.py both exist there; `accelerate` and `python` do not, and the
    node PATH does not include them either -- hence sourcing $ENV_FILE.
  * The upload path mapping is exactly the drive-strip rule node_path()
    assumed: C:\\Users\\... arrives as /Users/... on the node.
  * $MODEL_BASE_HOME holds both the diffusers tree and the 6.9 GB
    sd_xl_base_1.0.safetensors, so kohya can be pointed straight at it.

**Settled by env probe job 00010 (2026-08-28):** sourcing $ENV_FILE works
(rc=0) and puts 214 entries on PATH, including:
  * `python` / `python3` -> the package's own venv, **Python 3.11.9**
  * `accelerate` -> py-accelerate 0.33.0
  * `torchrun` -> py-torch 2.5.1
So DEFAULT_LAUNCHER = "python" is correct and present. `accelerate launch` is
also available if multi-GPU is ever wanted -- left unused because it can prompt
for a config on first run, and kohya's Accelerator falls back to single-process
under plain python anyway.

Nothing about the node environment is guessed at any more.
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

import cantemo_client as cantemo
import conductor_client as conductor

_logger = logging.getLogger(__name__)

# Package ids read from the live Conductor package list on 2026-08-27.
# Resolved by product name at run time; these are the fallback/reference values.
KOHYA_HOST_PRODUCT = "kohya"
MODEL_PRODUCTS = {
    "sdxl": "sdxl1-kohya",
    "flux-schnell": "fluxschnell-kohya",
    "flux-dev": "fluxdev-kohya",
    "sd35-large": "sd35l-kohya",
    "sd35-medium": "sd35m-kohya",
    "sd3-medium": "sd3m-kohya",
}

# RTX A5000 (24 GB) is the largest single-GPU card on the account; the 8 GB
# RTX 4000 types are not viable for SDXL LoRA training.
DEFAULT_INSTANCE_TYPE = "cw-epycmilan-4-rtxa5000-1"
DEFAULT_PROJECT = "TestProject"

# Relation types used for provenance edges in Cantemo. "unknown" is the API
# default; these are ours, and are what makes the round trip auditable in the
# MAM rather than only in Conductor's job history.
REL_TRAINED_FROM = "lora_trained_from"
REL_GENERATED_WITH = "generated_with_lora"

PROVENANCE_GROUP = "AI Provenance"

def default_workdir() -> str:
    """
    Where training images are staged before upload.

    This is not a cosmetic default. Conductor uploads preserve the path, and the
    render node runs tasks as the user `conductor` -- so a file staged under
    /root lands at /root on the node and the trainer cannot read its own dataset.
    That is exactly how job 00014 died: submitted from a container running as
    root, it failed with

        PermissionError: [Errno 13] Permission denied:
          '/root/.conductor-lora/aston-martin-f1-livery-v1/dataset'

    while the identical code submitted from a laptop (home = a Windows user
    directory, node path /Users/<name>/...) succeeded. The difference was never
    the code.

    So: use the home directory when it is a normal one, and fall back to a
    neutral path when running as root. Fixing this here rather than with an
    environment variable means a fresh container cannot reintroduce it by being
    deployed without the variable set.

    LORA_WORKDIR still overrides, for the case this does not anticipate.
    """
    override = os.getenv("LORA_WORKDIR")
    if override:
        return override
    home = os.path.expanduser("~")
    # A root home is the trap. Anything else -- a laptop, a normal service user
    # -- maps to a node path the `conductor` user can read.
    if not home or home in ("/root", "/") or home.startswith("/root/"):
        return "/Users/samsyn/.conductor-lora"
    return os.path.join(home, ".conductor-lora")


# Where AI outputs are filed in the MAM. Deliberately NOT the collection the
# training set is read from -- see cantemo_client.ensure_collection.
WORKBENCH_COLLECTION = "AI Workbench"
LORA_MODELS_COLLECTION = "LoRA Models"
LORA_OUTPUT_COLLECTION = "LoRA Output"

# Transcode profile requested when media is attached.
#
# Cantemo's import takes `tags` -- "a comma-separated list of transcode profile
# names to transcode". We were passing none, so nothing was transcoded, and
# because the POSTER is produced by the transcode, generated images landed with
# real 1024x1024 PNG media and NO thumbnail: a wall of grey placeholders in the
# grid, for assets that were perfectly sound.
#
# `lowres` is the profile this Portal actually uses -- a real broadcast asset
# (VX-1717) carries shapes `original` (MXF) and `lowres` (MP4), so the name is
# read off the live system rather than guessed. Configurable because it is a
# per-Portal name and the next MAM will call it something else.
TRANSCODE_TAGS = os.getenv("CANTEMO_TRANSCODE_TAGS", "lowres")

# Job states as they appear on the Cantemo item, so the whole round trip can be
# watched from the MAM without anyone opening Conductor's dashboard.
STATUS_SUBMITTED = "submitted"
STATUS_RUNNING = "training"
STATUS_GENERATING = "generating"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

# Conductor job states that mean "stop polling".
TERMINAL_STATES = {"success", "failed", "killed", "complete", "completed"}


# --- Conductor package resolution ----------------------------------------

async def resolve_packages(model: str = "sdxl") -> dict:
    """
    Find the kohya host package and the requested model plugin, and merge their
    environments the way Conductor would on the node.

    Returns {package_ids, environment, model_product, model_env}.
    """
    product = MODEL_PRODUCTS.get(model)
    if not product:
        raise ValueError(f"Unknown model {model!r}. Options: {', '.join(sorted(MODEL_PRODUCTS))}")

    raw = await conductor.list_software_packages()
    packages = raw if isinstance(raw, list) else (raw.get("data") or [])

    def latest(prod: str) -> Optional[dict]:
        matches = [p for p in packages if isinstance(p, dict) and p.get("product") == prod]
        if not matches:
            return None
        # Package names carry the date in major/minor/release; sort on the tuple
        # so 2025.07.16 beats 2025.03.19 rather than relying on list order.
        return sorted(
            matches,
            key=lambda p: (
                str(p.get("major_version", "")),
                str(p.get("minor_version", "")),
                str(p.get("release_version", "")),
            ),
        )[-1]

    host = latest(KOHYA_HOST_PRODUCT)
    plugin = latest(product)
    if not host:
        raise RuntimeError(f"No {KOHYA_HOST_PRODUCT} package on this Conductor account")
    if not plugin:
        raise RuntimeError(f"No {product} package on this Conductor account")

    env: dict = {}
    for pkg in (host, plugin):
        for entry in pkg.get("environment") or []:
            name, value = entry.get("name"), entry.get("value")
            if not name:
                continue
            if entry.get("merge_policy") == "append" and name in env:
                env[name] = f"{env[name]}:{value}"
            else:
                env[name] = value

    return {
        "package_ids": [host["package_id"], plugin["package_id"]],
        "environment": env,
        "model_product": product,
        "host_path": host.get("path"),
        "model_env": {k: v for k, v in env.items() if "MODEL" in k or "ENCODER" in k},
    }


# --- dataset staging ------------------------------------------------------

async def stage_training_images(
    item_ids: list[str],
    workdir: str,
    trigger_word: str,
    class_word: str = "style",
    repeats: int = 10,
) -> dict:
    """
    Pull the selected Cantemo items down and lay them out the way kohya expects.

    kohya reads a dataset directory whose subfolder name encodes the repeat
    count and the caption tokens: "<repeats>_<trigger> <class>". Everything in
    that folder is training data.

    Returns {dataset_dir, files, skipped} -- skipped lists items that had no
    resolvable media rather than failing the whole batch, so one bad asset in a
    demo selection does not sink the run.
    """
    dataset_root = os.path.join(workdir, "dataset")
    subdir = os.path.join(dataset_root, f"{int(repeats)}_{trigger_word} {class_word}")
    os.makedirs(subdir, exist_ok=True)

    files: list[str] = []
    skipped: list[dict] = []
    for item_id in item_ids:
        try:
            formats = await cantemo.get_formats(item_id)
            shapes = (formats or {}).get("formats") or []
            original = next((s for s in shapes if s.get("name") == "original"), None) or (
                shapes[0] if shapes else None
            )
            if not original:
                skipped.append({"item_id": item_id, "reason": "no shapes"})
                continue
            mime = str(original.get("mimeType") or "")
            if not mime.startswith("image/"):
                # .ai/.psd and video items are not trainable as-is. Extracting a
                # frame from video is a separate job, deliberately not silent here.
                skipped.append({"item_id": item_id, "reason": f"not an image ({mime or 'unknown'})"})
                continue
            ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
            dest = os.path.join(subdir, f"{item_id}{ext}")

            # Retry: job 00012 trained on 5 images instead of 8 because three
            # downloads hit transient connection failures and were silently
            # recorded as skipped. A flaky link should not quietly shrink the
            # training set — that changes the result without anyone noticing.
            got = None
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    got = await cantemo.download_to(item_id, dest, shape_id=original.get("id"))
                    if got:
                        break
                except Exception as exc:
                    last_err = exc
                    _logger.warning("[lora] %s download attempt %d failed: %s", item_id, attempt + 1, exc)
                await asyncio.sleep(2 * (attempt + 1))

            if got:
                files.append(got)
            else:
                reason = f"download failed after 3 tries ({type(last_err).__name__})" if last_err else "download failed"
                skipped.append({"item_id": item_id, "reason": reason})
        except Exception as exc:  # one bad asset must not sink the batch
            skipped.append({"item_id": item_id, "reason": f"{type(exc).__name__}: {exc}"})

    return {"dataset_dir": dataset_root, "subdir": subdir, "files": files, "skipped": skipped}


# --- job construction -----------------------------------------------------

def node_path(local_path: str) -> str:
    """
    Translate a local staging path into the path the same file will have on the
    render node.

    Conductor uploads preserve the directory structure but the nodes are Linux,
    so a Windows drive letter cannot survive: C:\\Users\\x\\dataset becomes
    /Users/x/dataset. Getting this wrong is silent -- the job submits happily and
    then fails on the node with "dataset directory not found" -- which is exactly
    how it was caught here, by dry-running the built command rather than reading
    it back from the code.

    CONFIRMED by discovery job 00005 (2026-08-27): a probe file uploaded from
    C:\\Users\\mktur\\AppData\\Local\\Temp\\conductor-lora-discovery\\ was found
    on the node at /Users/mktur/AppData/Local/Temp/conductor-lora-discovery/ --
    exactly this rule. Conductor does the mapping through its own path helper
    (CONDUCTOR_PATHHELPER=1, LD_PRELOAD=cio_path_helper.so,
    __conductor_letter_drives__=1 in the node environment).

    Override with the node_dataset_dir argument if a future case differs.
    """
    p = local_path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":  # drive-lettered Windows path
        p = p[2:]
    return p if p.startswith("/") else "/" + p



# Corrected against the real package listing from discovery job 00005
# (2026-08-27). What that run established:
#
#   * $KOHYA_PATH is the kohya_ss/sd-scripts REPO ROOT, not a bin directory of
#     executables. sdxl_train_network.py, sdxl_gen_img.py, train_network.py and
#     the networks/ package all sit directly in it.
#   * There is NO `accelerate` and NO `python` executable inside it, and the
#     node's PATH is only /usr/local/bin:/usr/bin:/bin:... -- so the original
#     "{kohya_bin}/accelerate launch" was wrong twice over.
#   * The package ships $ENV_FILE (/opt/kohya_2025.07.16/pkg_env.sh), which is
#     how a Spack-style layout like this puts its interpreter and PYTHONPATH on
#     PATH. Every command therefore sources it first.
#
# LAUNCHER stays configurable because whether `accelerate` exists after sourcing
# pkg_env.sh is still unconfirmed -- run the env probe to settle it. Plain
# `python` is the safe default: kohya's Accelerator falls back to single-process
# when not launched under accelerate, which is what a single-GPU job wants.
DEFAULT_LAUNCHER = "python"

#
# Two things this template learned from training job 00011, which failed after
# ~2.5 minutes on the node with no outputs at all:
#
# 1. **cd into $KOHYA_PATH first.** `--network_module=networks.lora` is a Python
#    module living inside the sd-scripts repo. Invoking the script by absolute
#    path from /tmp leaves `networks` unimportable — the single most likely
#    reason 00011 died early.
# 2. **Capture the log to output_path and always exit 0.** Conductor's task-log
#    API is broken on this account AND a failed job syncs no outputs, so a
#    failure is invisible twice over. Writing train.log next to the weights, and
#    exiting 0 so the sync happens, makes the next failure self-diagnosing. The
#    real exit code goes to train_status.txt — SUCCESS is therefore judged by
#    "did a .safetensors appear", which is what the ingest step already checks.
TRAIN_COMMAND_TEMPLATE = (
    "bash -c '"
    "set +e; mkdir -p {output_path}; "
    'source "$ENV_FILE"; '
    'cd "$KOHYA_PATH" || exit 0; '
    "{{ {launcher} sdxl_train_network.py"
    ' --pretrained_model_name_or_path="$MODEL_BASE_HOME"'
    " --train_data_dir={dataset_dir}"
    " --output_dir={output_path}"
    " --output_name={output_name}"
    " --network_module=networks.lora"
    " --network_dim={network_dim}"
    " --learning_rate={learning_rate}"
    " --max_train_epochs={epochs}"
    " --resolution={resolution}"
    " --save_model_as=safetensors"
    " --mixed_precision=bf16"
    " --cache_latents"
    # ---- VRAM, learned from job 00012 ----
    # 00012 got into the training loop and then died with CUDA OOM: 23.59 of
    # 23.68 GiB in use on the A5000. The traceback named the cause — it was in
    # sdxl_original_unet._attention doing a full `attention_scores.softmax`,
    # i.e. the naive attention path, because no memory-efficient backend was
    # requested. Three flags fix it, cheapest first:
    #   --sdpa                    torch 2.5's built-in memory-efficient
    #                             attention. No xformers package needed, and
    #                             this alone removes the quadratic buffer.
    #   --gradient_checkpointing  trades compute for activation memory.
    #   --network_train_unet_only skips the two text encoders entirely — a
    #                             large saving, and irrelevant to a livery/style
    #                             LoRA, which learns appearance not vocabulary.
    " --sdpa"
    " --gradient_checkpointing"
    " --network_train_unet_only; "
    "echo train_rc=$? > {output_path}/train_status.txt; "
    "}} > {output_path}/train.log 2>&1; "
    "ls -la {output_path} >> {output_path}/train.log 2>&1; "
    "exit 0'"
)

# Probes what the package environment actually provides once sourced -- the one
# thing discovery 00005 could not answer, because it never sourced $ENV_FILE.
#
# Job 00008 (the first attempt) FAILED on the node. A probe whose whole purpose
# is to report what is missing must never fail because something is missing:
# `set +e` so a absent binary doesn't abort, no `-l` (a login shell sources
# profile scripts that can exit non-zero), `source` guarded, and an explicit
# `exit 0` so Conductor always sees success and keeps the log.
#
# Writes its findings to a FILE in output_path rather than to stdout.
#
# Conductor's task-log API is broken on this account (/get_log_file 500s for
# every parameter shape), so a probe that only prints needs a human in the
# dashboard for every iteration -- which is exactly what made jobs 00008 and
# 00009 expensive to diagnose. Output files, by contrast, come back through
# get_job_outputs as signed URLs we can fetch directly. Same trick works for
# any future on-node debugging.
#
# Structural care, learned from those two failures: no login shell (profile
# scripts can exit non-zero), the source runs in a SUBSHELL so an `exit` inside
# pkg_env.sh cannot kill the probe, no `..` in paths (the node's LD_PRELOAD path
# helper flags traversal), no piping into `head` (SIGPIPE), and an explicit
# `exit 0` so Conductor records success and keeps the outputs.
ENV_PROBE_COMMAND = (
    "bash -c '"
    "set +e; mkdir -p /lora_discovery; { "
    'echo ===BEFORE===; echo "PATH=$PATH"; '
    'echo ===ENVFILE===; cat "$ENV_FILE"; '
    "echo ===SOURCED_SUBSHELL===; "
    '( . "$ENV_FILE" >/dev/null 2>&1; echo "source_rc=$?"; echo "PATH_AFTER=$PATH"; '
    "for b in python python3 accelerate torchrun pip; do "
    'printf "%s -> " "$b"; command -v "$b" || echo MISSING; done; '
    "python -V 2>&1; python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1 ); "
    'echo ===KOHYA_BIN===; ls -la "$KOHYA_PATH"; '
    "echo ===DONE===; "
    "} > /lora_discovery/probe.txt 2>&1; exit 0'"
)


DISCOVERY_COMMAND = (
    "bash -lc '"
    "echo ===ENV===; env | sort; "
    "echo ===KOHYA_BIN===; ls -la \"$KOHYA_PATH\" 2>&1 | head -60; "
    "echo ===TRAIN_SCRIPTS===; "
    "find /opt/kohya_* -maxdepth 6 -name \"*train_network*\" -o -maxdepth 6 -name \"accelerate\" "
    "-o -maxdepth 6 -name \"python*\" 2>/dev/null | head -40; "
    "echo ===PKG_TREE===; find /opt/kohya_* -maxdepth 4 -type d 2>/dev/null | head -60; "
    "echo ===MODEL===; ls -la \"$MODEL_BASE_HOME\" 2>&1 | head -20; "
    "echo ===UPLOAD_LOCATION===; "
    "find / -name \"DISCOVERY_PROBE.txt\" -not -path \"/proc/*\" 2>/dev/null | head -5; "
    "echo ===CWD===; pwd; ls -la .; "
    "echo ===DONE==="
    "'"
)


async def build_discovery_job(
    probe_file: str,
    project: str = DEFAULT_PROJECT,
    instance_type: str = "cw-xeonv3-4",
) -> dict:
    """
    A deliberately tiny job whose only purpose is to tell us two things we are
    currently guessing at: what is actually inside the kohya package (entry
    point names, python location, sd-scripts layout), and where an uploaded
    file lands on the node.

    `probe_file` is a small local file uploaded with the job purely so the
    ===UPLOAD_LOCATION=== section can find it and reveal the path mapping --
    which is the thing node_path() is currently assuming.

    Cheapest CPU instance, no GPU, no model load. Packages are still mounted so
    the environment variables resolve.
    """
    packages = await resolve_packages("sdxl")
    return {
        "job_title": "kohya package discovery (listing only)",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": packages["package_ids"],
        "upload_only": False,
        "force": False,
        "local_upload": True,
        "preemptible": True,
        "output_path": "/lora_discovery",
        "environment": packages["environment"],
        "upload_paths": [probe_file],
        "scout_frames": "1",
        "tasks_data": [{"command": DISCOVERY_COMMAND, "frames": "1"}],
    }


async def build_env_probe_job(
    project: str = DEFAULT_PROJECT,
    instance_type: str = "cw-xeonv3-4",
) -> dict:
    """
    Second, smaller discovery: source $ENV_FILE and report what lands on PATH.

    Discovery 00005 listed the package but never sourced its environment file,
    so it could not say how to invoke python. This settles DEFAULT_LAUNCHER.
    """
    packages = await resolve_packages("sdxl")
    return {
        "job_title": "kohya env probe (source pkg_env.sh)",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": packages["package_ids"],
        "upload_only": False,
        "force": False,
        "local_upload": True,
        "preemptible": True,
        "output_path": "/lora_discovery",
        "environment": packages["environment"],
        "upload_paths": [],
        "scout_frames": "1",
        "tasks_data": [{"command": ENV_PROBE_COMMAND, "frames": "1"}],
    }


async def submit_env_probe(dry_run: bool = True, **kwargs) -> dict:
    """Submit the env probe. Cheap, CPU-only, but still real money."""
    job_args = await build_env_probe_job(**kwargs)
    if dry_run:
        return {"dry_run": True, "job_args": job_args}
    import ciocore.conductor_submit

    result, code = await asyncio.to_thread(
        lambda: ciocore.conductor_submit.Submit(job_args).main()
    )
    return {"dry_run": False, "jid": result.get("jid"), "code": code, "status": result.get("status")}


async def submit_discovery(probe_file: str, dry_run: bool = True, **kwargs) -> dict:
    """Submit the discovery job. Cheap, but still real money, so still opt-in."""
    job_args = await build_discovery_job(probe_file, **kwargs)
    if dry_run:
        return {"dry_run": True, "job_args": job_args}
    import ciocore.conductor_submit

    result, code = await asyncio.to_thread(
        lambda: ciocore.conductor_submit.Submit(job_args).main()
    )
    return {"dry_run": False, "jid": result.get("jid"), "code": code, "status": result.get("status")}


async def build_training_job(
    item_ids: list[str],
    label: str,
    workdir: str,
    model: str = "sdxl",
    trigger_word: str = "sks",
    class_word: str = "style",
    project: str = DEFAULT_PROJECT,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    epochs: int = 10,
    network_dim: int = 32,
    learning_rate: str = "1e-4",
    resolution: str = "1024,1024",
    repeats: int = 10,
    preemptible: bool = False,
    node_dataset_dir: Optional[str] = None,
    launcher: str = DEFAULT_LAUNCHER,
) -> dict:
    """
    Assemble everything Conductor needs for one LoRA training job.

    Pure construction plus the local staging -- submits nothing. Returns the
    ciocore job_args alongside the staging report so a caller (or a dry run)
    can inspect exactly what would be sent.

    preemptible defaults False here, unlike renders: a preempted training job
    restarts from scratch and a demo cannot absorb that.
    """
    if not item_ids:
        raise ValueError("No items selected for training")

    os.makedirs(workdir, exist_ok=True)
    packages = await resolve_packages(model)
    staged = await stage_training_images(
        item_ids, workdir, trigger_word=trigger_word, class_word=class_word, repeats=repeats
    )
    if not staged["files"]:
        raise RuntimeError(
            "No trainable images among the selected items: "
            + json.dumps(staged["skipped"])
        )

    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label).strip("-").lower()
    output_path = f"/lora_training/{slug}"
    env = dict(packages["environment"])
    # The training command runs on a Linux node, so it must NOT be handed the
    # local staging path -- see node_path().
    node_dataset_dir = node_dataset_dir or node_path(staged["dataset_dir"])

    # The OOM in job 00012 recommended this itself; it costs nothing and reduces
    # allocator fragmentation, which is what turns a near-miss into a failure.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    command = TRAIN_COMMAND_TEMPLATE.format(
        launcher=launcher,
        dataset_dir=node_dataset_dir,
        output_path=output_path,
        output_name=slug,
        network_dim=network_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        resolution=resolution,
    )

    job_args = {
        "job_title": f"LoRA train -- {label}",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": packages["package_ids"],
        "upload_only": False,
        "force": False,
        "local_upload": True,  # the dataset lives on this machine, not on a mounted store
        "preemptible": preemptible,
        "autoretry_policy": {"preempted": {"max_retries": 1}},
        "output_path": output_path,
        "environment": env,
        "upload_paths": staged["files"],
        "scout_frames": "1",
        "tasks_data": [{"command": command, "frames": "1"}],
        "metadata": {
            "source_item_ids": ",".join(item_ids),
            "label": label,
            "base_model": packages["model_product"],
            "trigger_word": trigger_word,
        },
    }
    return {"job_args": job_args, "staged": staged, "packages": packages, "output_path": output_path}


async def submit_training(dry_run: bool = True, **kwargs) -> dict:
    """
    Build and (optionally) submit a LoRA training job.

    dry_run=True by default -- this spends real GPU money on Mark's Conductor
    account, so submission is always an explicit opt-in.
    """
    built = await build_training_job(**kwargs)
    if dry_run:
        return {
            "dry_run": True,
            "job_args": built["job_args"],
            "staged_files": len(built["staged"]["files"]),
            "skipped": built["staged"]["skipped"],
            "output_path": built["output_path"],
        }

    import ciocore.conductor_submit  # imported late: absent in lean deployments

    result, code = await asyncio.to_thread(
        lambda: ciocore.conductor_submit.Submit(built["job_args"]).main()
    )
    return {
        "dry_run": False,
        "jid": result.get("jid"),
        "code": code,
        "status": result.get("status"),
        "output_path": built["output_path"],
        "staged_files": len(built["staged"]["files"]),
        "skipped": built["staged"]["skipped"],
    }


# --- inference ------------------------------------------------------------

# Same caveat as TRAIN_COMMAND_TEMPLATE: this is the expected shape, not a
# proven one. kohya's sd-scripts ships sdxl_gen_img.py alongside the trainer,
# which is why inference can reuse exactly the same two packages -- no second
# software stack, and the base model is again already on the node.
# Same shape as training, and for the same two reasons — sdxl_gen_img.py also
# imports networks.lora, and a failed generation is just as invisible.
# The prompt travels in the ENVIRONMENT, not in the command string.
#
# It used to be interpolated into the command. That cannot work: the whole thing
# is wrapped in bash -c '...', and inside SINGLE quotes a backslash is a literal
# character, not an escape. So the node received
#
#     --prompt \"amf1 livery, studio lighting, three-quarter front view...
#
# bash split that on spaces, sdxl_gen_img.py took the first word as the prompt
# and died with "error: unrecognized arguments: livery, studio lighting, ...".
# Job 00015 reported SUCCESS and produced no images -- the wrapper's exit code
# is not the generator's, exactly the trap job 00014 set for training.
#
# Passing it through the environment removes the quoting problem instead of
# escaping around it: a prompt cannot break out of a shell command it never
# appears in. Apostrophes and punctuation work now as a side effect.
INFER_COMMAND_TEMPLATE = (
    "bash -c '"
    "set +e; mkdir -p {output_path}; "
    'source "$ENV_FILE"; '
    'cd "$KOHYA_PATH" || exit 0; '
    "{{ {launcher} sdxl_gen_img.py"
    ' --ckpt "$MODEL_BASE_HOME"'
    " --network_module networks.lora"
    " --network_weights {lora_path}"
    " --outdir {output_path}"
    ' --prompt "$LORA_PROMPT"'
    " --images_per_prompt {count}"
    " --W {width} --H {height}"
    " --steps {steps}"
    " --seed {seed}; "
    "echo infer_rc=$? > {output_path}/infer_status.txt; "
    "}} > {output_path}/infer.log 2>&1; "
    "exit 0'"
)


async def stage_lora_from_cantemo(lora_item_id: str, workdir: str) -> dict:
    """
    Pull a LoRA's .safetensors back out of the MAM so it can be uploaded to the
    render node.

    The round trip is deliberately symmetric: the LoRA lives in Cantemo as a
    first-class asset, so running inference means fetching it from the MAM like
    any other piece of media -- not reaching into Conductor's job outputs for a
    second time.
    """
    os.makedirs(workdir, exist_ok=True)
    formats = await cantemo.get_formats(lora_item_id)
    shapes = (formats or {}).get("formats") or []
    if not shapes:
        raise RuntimeError(f"LoRA item {lora_item_id} has no downloadable shape")
    original = next((s for s in shapes if s.get("name") == "original"), shapes[0])
    dest = os.path.join(workdir, f"{lora_item_id}.safetensors")
    got = await cantemo.download_to(lora_item_id, dest, shape_id=original.get("id"))
    if not got:
        raise RuntimeError(f"Could not download LoRA weights for {lora_item_id}")
    return {"path": got, "shape_id": original.get("id")}


async def build_inference_job(
    lora_item_id: str,
    prompt: str,
    workdir: str,
    model: str = "sdxl",
    count: int = 4,
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    seed: int = 42,
    project: str = DEFAULT_PROJECT,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    preemptible: bool = True,
    node_lora_path: Optional[str] = None,
    launcher: str = DEFAULT_LAUNCHER,
) -> dict:
    """
    Assemble a generation job that applies a MAM-held LoRA to its base model.

    preemptible defaults True here, unlike training: inference is short and
    cheap to retry, so spot capacity is the sensible default.
    """
    if not prompt.strip():
        raise ValueError("A prompt is required")

    packages = await resolve_packages(model)
    staged = await stage_lora_from_cantemo(lora_item_id, workdir)

    env = dict(packages["environment"])
    output_path = f"/lora_inference/{lora_item_id}"

    # Only newlines are stripped now -- they would end the command line itself.
    # Quotes and apostrophes are safe, because the prompt rides in the
    # environment rather than being interpolated into a shell string.
    safe_prompt = " ".join(prompt.split())
    env["LORA_PROMPT"] = safe_prompt

    command = INFER_COMMAND_TEMPLATE.format(
        launcher=launcher,
        lora_path=node_lora_path or node_path(staged["path"]),
        output_path=output_path,
        count=count,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
    )

    job_args = {
        "job_title": f"LoRA generate -- {safe_prompt[:50]}",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": packages["package_ids"],
        "upload_only": False,
        "force": False,
        "local_upload": True,
        "preemptible": preemptible,
        "autoretry_policy": {"preempted": {"max_retries": 1}},
        "output_path": output_path,
        "environment": env,
        "upload_paths": [staged["path"]],
        "scout_frames": "1",
        "tasks_data": [{"command": command, "frames": "1"}],
        "metadata": {
            "lora_item_id": lora_item_id,
            "prompt": safe_prompt,
            "base_model": packages["model_product"],
        },
    }
    return {"job_args": job_args, "staged": staged, "output_path": output_path, "prompt": safe_prompt}


async def submit_inference(dry_run: bool = True, **kwargs) -> dict:
    """Build and (optionally) submit a generation job. Opt-in to spending, as ever."""
    built = await build_inference_job(**kwargs)
    if dry_run:
        return {
            "dry_run": True,
            "job_args": built["job_args"],
            "output_path": built["output_path"],
            "prompt": built["prompt"],
        }

    import ciocore.conductor_submit

    result, code = await asyncio.to_thread(
        lambda: ciocore.conductor_submit.Submit(built["job_args"]).main()
    )
    return {
        "dry_run": False,
        "jid": result.get("jid"),
        "code": code,
        "status": result.get("status"),
        "output_path": built["output_path"],
        "prompt": built["prompt"],
    }


# --- write-back -----------------------------------------------------------

# Logical name -> Cantemo field id.
#
# Cantemo enforces a field-id format of {custom-name}_{field-name}, each part
# starting with a letter, so a bare "label" or "prompt" is rejected outright.
# Prefixing every id with aiprov_ satisfies that and makes collisions with the
# Portal's existing fields impossible -- these are global, not group-scoped.
#
# These strings must match the field ids created in the Portal admin UI exactly.
# Change them here and in the UI together, or writes will silently drop fields.
#
# Reduced to the EIGHT fields that exist on the Portal (created 2026-08-27).
# Dropped deliberately: created_by / created_at / generated_by / generated_at
# (Cantemo stamps creating user and date natively), and lora_used (the
# generated_with_lora relation edge says it better). The two job-id fields
# collapsed into one aiprov_job_id.
#
# HARD REQUIREMENT, learned the hard way: a field must be ATTACHED TO THE GROUP,
# not merely exist. Writing an orphan field id returns
# 400 "Vidispine error: notFound metadata-field <id>".
#
# VERIFIED 2026-08-28: all eight write and read back on a scratch item, 8/8.
#
# Note prompt -> prov_prompt2, not prov_prompt. The id was taken during an
# earlier attempt, so the field that actually made it into the group carries the
# 2. Renaming it in the Portal would mean another pass through the group
# builder for no functional gain; the mapping layer exists precisely so an
# awkward external id stays external.
PROVENANCE_FIELD_IDS = {
    "provenance_kind": "prov_kind",
    "status": "prov_status",
    "label": "prov_label",
    "base_model": "prov_base_model",
    "trigger_word": "prov_trigger_word",
    "prompt": "prov_prompt2",
    "source_asset_ids": "prov_source_assets",
    "job_id": "prov_job_id",
    # Who ran it. Added by Mark in the Portal admin UI on 2026-08-31 -- field
    # CREATION and group ATTACHMENT are both admin-UI-only on Cantemo (no v2
    # route, and Vidispine's metadataelement path does not attach).
    #
    # The "3" is not a typo and must not be "tidied" back to the bare name.
    # Cantemo will not reuse a field id, so a half-finished creation burns it:
    # `prov_created_by` and `prov_created_by2` both exist on the group as dead
    # earlier attempts, exactly as `prov_prompt2` did before. **prov_created_by3
    # is the live one** -- verified by writing and reading it back on VX-4422.
    "created_by": "prov_created_by3",
}

# Cached list of field ids actually attached to the provenance group on THIS
# Portal. Writing a field that exists but is not attached to the group returns
# "400 notFound metadata-field", so presence has to mean "attached", which is
# exactly what get_metadata_group reports.
_group_fields_cache: Optional[set[str]] = None


async def provenance_group_fields(refresh: bool = False) -> set[str]:
    global _group_fields_cache
    if _group_fields_cache is not None and not refresh:
        return _group_fields_cache
    try:
        group = await cantemo.get_metadata_group(PROVENANCE_GROUP)
        _group_fields_cache = {
            str(f.get("name")) for f in (group or {}).get("fields", []) if f.get("name")
        }
    except Exception as exc:
        _logger.error("[lora] could not read provenance group: %s", exc)
        _group_fields_cache = set()
    return _group_fields_cache


async def _writable_provenance_fields(values: dict) -> tuple[list[dict], list[str]]:
    """
    Translate provenance values into fields this Portal will actually accept.

    Two different failure modes, deliberately treated differently:

      * An UNMAPPED key is a bug in our code -- raise, because a provenance
        record that quietly loses a field is worse than one that fails.
      * A MAPPED key whose field is not attached to the group on this Portal is
        an install difference, not a bug. Writing it would 400 the whole
        metadata call and lose the other seven fields with it. So it is skipped
        and REPORTED, and it starts working by itself once someone adds the
        field in the admin UI -- no code change, no redeploy.

    Returns (fields, skipped_logical_names).
    """
    available = await provenance_group_fields()
    fields: list[dict] = []
    skipped: list[str] = []
    for key, value in values.items():
        if value is None or value == "":
            continue
        field_id = PROVENANCE_FIELD_IDS.get(key)
        if not field_id:
            raise KeyError(f"No Cantemo field id mapped for provenance key {key!r}")
        # An empty available-set means we could not read the group at all --
        # attempt the write rather than skipping everything on a transient read
        # failure.
        if available and field_id not in available:
            skipped.append(f"{key} ({field_id})")
            continue
        fields.append({"name": field_id, "value": str(value)})
    return fields, skipped




async def ingest_lora_to_cantemo(
    job_id: str,
    label: str,
    source_item_ids: list[str],
    base_model: str,
    trigger_word: str,
    created_by: str,
    notranscode: bool = True,
    collection: Optional[str] = LORA_MODELS_COLLECTION,
) -> dict:
    """
    Pull the trained .safetensors off Conductor and land it in the MAM as a new
    item, related back to every asset it was trained on.

    notranscode defaults True: a LoRA is model weights, and asking the Portal to
    make video proxies of it is meaningless.
    """
    outputs = await conductor.get_job_outputs(job_id)
    weights = [
        f
        for task in outputs.get("downloads", [])
        for f in task.get("files", [])
        if output_name(f).endswith(".safetensors")
    ]
    if not weights:
        # "No .safetensors" is a symptom. The cause is in the job's own log, and
        # we already have the outputs in hand -- so say what actually happened
        # rather than making the next person go and find it.
        detail = await training_failure_detail(outputs)
        produced = sorted(
            output_name(f)
            for task in outputs.get("downloads", [])
            for f in task.get("files", [])
        )
        error = f"Training job {job_id} produced no weights"
        if detail:
            error += f". The trainer reported: {detail}"
        return {
            "ok": False,
            "error": error,
            "job_id": job_id,
            "produced_files": produced,
            "outputs": outputs,
        }

    weight = weights[0]
    url = weight.get("url") or weight.get("signed_url")
    if not url:
        return {"ok": False, "error": "Output file carried no signed URL", "file": weight}

    item = await cantemo.create_placeholder(title=f"LoRA -- {label}")
    item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
    if not item_id:
        return {"ok": False, "error": "Could not read new item id", "response": item}

    await cantemo.import_uri(item_id, url, notranscode=notranscode)
    prov_fields, prov_skipped = await _writable_provenance_fields(
        {
            "provenance_kind": "lora",
            "label": label,
            "base_model": base_model,
            "trigger_word": trigger_word,
            "job_id": job_id,
            "source_asset_ids": ",".join(source_item_ids),
            "created_by": created_by,
        }
    )
    await cantemo.set_metadata(item_id, prov_fields, group_name=PROVENANCE_GROUP)

    # The edges are the point: every training asset becomes a traversable
    # relation, so the MAM itself can answer "what was this trained on".
    linked, failed = [], []
    for src in source_item_ids:
        try:
            await cantemo.create_relation(item_id, src, relation_type=REL_TRAINED_FROM)
            linked.append(src)
        except Exception as exc:
            failed.append({"item_id": src, "error": str(exc)})

    filed = await _file_into(collection, [item_id])

    return {
        "ok": True,
        "item_id": item_id,
        "job_id": job_id,
        "weights_file": output_name(weight),
        "related": linked,
        "relation_failures": failed,
        "filed": filed,
        "provenance_written": [f["name"] for f in prov_fields],
        "provenance_skipped": prov_skipped,
    }


async def _file_into(collection: Optional[str], item_ids: list[str]) -> Optional[dict]:
    """File freshly created items, never fatal -- the asset exists either way."""
    if not collection or not item_ids:
        return None
    try:
        coll_id = await cantemo.ensure_collection(collection, parent_name=WORKBENCH_COLLECTION)
        if not coll_id:
            return {"collection": collection, "error": "could not find or create"}
        await cantemo.add_to_collection(coll_id, item_ids)
        return {"collection": collection, "collection_id": coll_id, "filed": len(item_ids)}
    except Exception as exc:
        return {"collection": collection, "error": f"{type(exc).__name__}: {exc}"}


async def training_failure_detail(outputs: dict) -> Optional[str]:
    """
    Explain why a job that Conductor calls "success" produced no weights.

    The wrapper script's exit code is not the trainer's. kohya can die and the
    task still ends 0, so the job reads green while `output_path` holds nothing
    but a log. When that happens the ONLY honest report is what the trainer
    said, and it is already sitting in the outputs we just fetched.

    Job 00014 is the case this exists for: it reported success, produced only
    train.log and train_status.txt, and the caller said "No .safetensors in
    outputs" -- true, useless, and pointing at the wrong layer. The real cause
    was three lines down in train.log (a PermissionError on the dataset, from
    staging under /root when the render node runs tasks as `conductor`).

    Returns a human-readable cause, or None if the outputs say nothing useful.
    """
    files = [f for task in outputs.get("downloads", []) for f in task.get("files", [])]
    by_name = {output_name(f): f for f in files}

    async def fetch(name: str) -> str:
        f = by_name.get(name)
        url = (f or {}).get("url") or (f or {}).get("signed_url")
        if not url:
            return ""
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.text
        except Exception:
            return ""

    # Training writes train_*, inference writes infer_*. Looking for only one
    # set is why job 00015's failure went unexplained: the helper found no
    # train.log, said nothing, and the caller fell back to "produced no images"
    # -- the same useless symptom this function exists to replace.
    status = ""
    log = ""
    for prefix in ("train", "infer"):
        status = status or (await fetch(f"{prefix}_status.txt")).strip()
        log = log or await fetch(f"{prefix}.log")

    if not status and not log:
        return None

    # The last traceback line names the actual failure; the noise above it is
    # deprecation warnings nobody needs to read.
    cause = ""
    if log:
        lines = [ln.rstrip() for ln in log.splitlines() if ln.strip()]
        for ln in reversed(lines):
            # argparse failures do not raise -- the script prints usage and
            # exits 2. That is how a mis-quoted prompt looks, so it has to be
            # recognised alongside the exceptions.
            if ln.startswith("usage:") or ": error:" in ln:
                cause = ln
                break
            if any(ln.startswith(p) for p in ("PermissionError", "FileNotFoundError", "RuntimeError",
                                              "OSError", "ValueError", "AssertionError",
                                              "torch.cuda.OutOfMemoryError")):
                cause = ln
                break
        if not cause:
            cause = " / ".join(lines[-3:])

    parts = [p for p in (status or None, cause or None) if p]
    return " -- ".join(parts) if parts else None


def output_name(f: dict) -> str:
    """
    The filename of a Conductor output.

    Conductor calls this field **relative_path** — there is no `name` and no
    `path`. Filtering on those (as this module first did) silently matches
    nothing, so a job that produced a perfectly good .safetensors reports "no
    weights found". Cost a confused round trip on job 00013; keep all the
    candidates here so it cannot happen again.
    """
    return str(f.get("relative_path") or f.get("name") or f.get("output_path") or f.get("path") or "")


async def job_status(job_id: str) -> dict:
    """
    Conductor's view of a job, reduced to what the MAM needs to show.

    list_jobs' id-range filter does not actually filter (it returns job 1
    regardless), so match on the padded jid instead.
    """
    jobs = await conductor.list_jobs()
    data = jobs.get("data", jobs) if isinstance(jobs, dict) else jobs
    want = str(job_id).zfill(5)
    for job in (data if isinstance(data, list) else [data]):
        if str(job.get("jid")) == want:
            return {
                "jid": want,
                "status": job.get("status"),
                "running": job.get("running"),
                "success": job.get("success"),
                "failed": job.get("failed"),
                "pending": job.get("pending"),
                "terminal": str(job.get("status") or "").lower() in TERMINAL_STATES,
            }
    return {"jid": want, "status": "not_found", "terminal": False}


async def create_tracked_lora_item(
    label: str,
    job_id: str,
    source_item_ids: list[str],
    base_model: str,
    trigger_word: str,
    created_by: str,
) -> dict:
    """
    Create the LoRA's MAM item the moment training is submitted, not when it
    finishes.

    This is what keeps the demo inside Cantemo. The item exists immediately with
    status "submitted" and relation edges to every training asset, so the whole
    story -- what is being trained, from what, by whom, and how far along it is
    -- is visible in the MAM while the GPU work happens. Nobody needs to open
    Conductor's dashboard to see whether it worked.

    The media arrives later, via finalize_tracked_lora().
    """
    item = await cantemo.create_placeholder(title=f"LoRA -- {label}")
    item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
    if not item_id:
        return {"ok": False, "error": "Could not read new item id", "response": item}

    tracked_fields, _ = await _writable_provenance_fields(
        {
            "provenance_kind": "lora",
            "status": STATUS_SUBMITTED,
            "label": label,
            "base_model": base_model,
            "trigger_word": trigger_word,
            "job_id": job_id,
            "source_asset_ids": ",".join(source_item_ids),
        }
    )
    await cantemo.set_metadata(
        item_id,
        tracked_fields,
        group_name=PROVENANCE_GROUP,
    )

    linked, failed = [], []
    for src in source_item_ids:
        try:
            await cantemo.create_relation(item_id, src, relation_type=REL_TRAINED_FROM)
            linked.append(src)
        except Exception as exc:
            failed.append({"item_id": src, "error": str(exc)})

    return {"ok": True, "item_id": item_id, "job_id": job_id, "related": linked, "relation_failures": failed}


async def sync_status_to_mam(item_id: str, job_id: str, kind: str = "lora") -> dict:
    """
    Copy Conductor's job state onto the MAM item once.

    Call it on a timer (or from a rules-engine action) and the item's status
    field tracks the job. Deliberately one shot rather than a loop, so the
    caller decides the cadence and nothing blocks.
    """
    status = await job_status(job_id)
    raw = str(status.get("status") or "").lower()
    if raw in ("success", "complete", "completed"):
        mapped = STATUS_READY
    elif raw in ("failed", "killed"):
        mapped = STATUS_FAILED
    elif raw in ("running",):
        mapped = STATUS_GENERATING if kind == "generated_image" else STATUS_RUNNING
    else:
        mapped = STATUS_SUBMITTED

    await cantemo.set_metadata(
        item_id, (await _writable_provenance_fields({"status": mapped}))[0], group_name=PROVENANCE_GROUP
    )
    return {"item_id": item_id, "conductor_status": status.get("status"), "mam_status": mapped,
            "terminal": status.get("terminal")}


async def finalize_tracked_lora(item_id: str, job_id: str) -> dict:
    """
    Attach the trained weights to the item that has been tracking the job.

    Separate from create_tracked_lora_item so the item can exist (and be watched)
    for the whole training run, with the media arriving at the end.
    """
    outputs = await conductor.get_job_outputs(job_id)
    weights = [
        f
        for task in outputs.get("downloads", [])
        for f in task.get("files", [])
        if output_name(f).endswith(".safetensors")
    ]
    if not weights:
        await cantemo.set_metadata(
            item_id, (await _writable_provenance_fields({"status": STATUS_FAILED}))[0], group_name=PROVENANCE_GROUP
        )
        return {"ok": False, "error": f"No .safetensors in outputs of job {job_id}"}

    url = weights[0].get("url") or weights[0].get("signed_url")
    if not url:
        return {"ok": False, "error": "Output file carried no signed URL"}

    await cantemo.import_uri(item_id, url, notranscode=True)
    await cantemo.set_metadata(
        item_id, (await _writable_provenance_fields({"status": STATUS_READY}))[0], group_name=PROVENANCE_GROUP
    )
    return {"ok": True, "item_id": item_id, "weights_file": output_name(weights[0])}


async def stamp_source_assets(
    source_item_ids: list[str],
    label: str,
    job_id: str,
    base_model: str,
    trigger_word: str,
) -> dict:
    """
    Write provenance onto the TRAINING IMAGES themselves.

    Until this ran, a source still carried nothing: open one in the MAM and it
    said "There is no metadata for this item yet", even though it had helped
    train a model. The relation edge existed, but an operator looking at the
    asset saw an unexplained picture.

    Provenance should read in both directions. The LoRA says what it came from;
    each contributing image should say what it went into.

    No new fields are needed -- the same eight carry a third kind alongside
    "lora" and "generated_image":
        prov_kind = training_source, and prov_label / prov_job_id /
        prov_base_model / prov_trigger_word name the model it fed.
    prompt and source_assets stay empty; neither means anything on an original.

    NB this writes to assets we did not create. It is additive and reversible,
    but on someone else's Portal that is worth doing deliberately.
    """
    stamped, failed = [], []
    for item_id in source_item_ids:
        try:
            stamp_fields, _ = await _writable_provenance_fields(
                {
                    "provenance_kind": "training_source",
                    "status": STATUS_READY,
                    "label": label,
                    "base_model": base_model,
                    "trigger_word": trigger_word,
                    "job_id": job_id,
                }
            )
            await cantemo.set_metadata(item_id, stamp_fields, group_name=PROVENANCE_GROUP)
            stamped.append(item_id)
        except Exception as exc:
            failed.append({"item_id": item_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": not failed, "stamped": stamped, "failures": failed, "count": len(stamped)}


async def lora_identity(lora_item_id: str) -> dict:
    """
    Read a LoRA item's own provenance back out of the MAM.

    Generated images have to name the model that made them, and the caller
    should not have to remember what that model was called -- the MAM already
    knows. Reading it here rather than passing it in means the label on an image
    can never drift from the label on the LoRA it came from.

    Falls back to the item title (minus the "LoRA -- " prefix the ingest adds)
    when the provenance fields are not readable.
    """
    label, trigger = "", ""
    try:
        meta = await cantemo.get_metadata(lora_item_id)
        # The metadata document nests differently by Portal version, so walk it
        # rather than assuming one shape.
        def walk(node: Any) -> None:
            nonlocal label, trigger
            if isinstance(node, dict):
                name, value = node.get("name"), node.get("value")
                if name == "prov_label" and value:
                    label = str(value if not isinstance(value, list) else value[0])
                if name == "prov_trigger_word" and value:
                    trigger = str(value if not isinstance(value, list) else value[0])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(meta)
    except Exception as exc:
        _logger.error("[lora] could not read provenance of %s: %s", lora_item_id, exc)

    if not label:
        try:
            item = await cantemo.get_item(lora_item_id)
            title = item.get("title") or ""
            if isinstance(title, list):
                title = title[0] if title else ""
            label = str(title).replace("LoRA -- ", "").replace("LoRA — ", "").strip()
        except Exception:
            pass
    return {"label": label, "trigger_word": trigger}


async def ingest_generated_images(
    job_id: str,
    lora_item_id: str,
    prompt: str,
    base_model: str,
    created_by: str,
    collection: Optional[str] = LORA_OUTPUT_COLLECTION,
) -> dict:
    """
    Land inference output back in the MAM, each image related to the LoRA that
    made it and carrying the prompt that produced it.
    """
    outputs = await conductor.get_job_outputs(job_id)
    images = [
        f
        for task in outputs.get("downloads", [])
        for f in task.get("files", [])
        if output_name(f).lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not images:
        detail = await training_failure_detail(outputs)
        produced = sorted(
            output_name(f)
            for task in outputs.get("downloads", [])
            for f in task.get("files", [])
        )
        error = f"Inference job {job_id} produced no images"
        if detail:
            error += f". The generator reported: {detail}"
        return {"ok": False, "error": error, "job_id": job_id, "produced_files": produced}

    identity = await lora_identity(lora_item_id)
    skipped_report: list[str] = []

    created = []
    for idx, img in enumerate(images, start=1):
        url = img.get("url") or img.get("signed_url")
        if not url:
            continue
        item = await cantemo.create_placeholder(title=f"{prompt[:60]} ({idx})")
        item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
        if not item_id:
            continue
        # Ask for a proxy, which is what makes Cantemo generate the poster.
        await cantemo.import_uri(item_id, url, notranscode=False, tags=TRANSCODE_TAGS or None)
        # Everything needed to answer "what made this, from what, and who asked"
        # WITHOUT leaving the asset: the base model, the LoRA by name, its
        # trigger word, the full prompt, the person, and the compute job.
        # The relation edge below carries the same link as graph structure; the
        # metadata carries it as text, because a person reading the asset panel
        # should not have to traverse anything.
        prov_fields, prov_skipped = await _writable_provenance_fields(
            {
                "provenance_kind": "generated_image",
                "label": identity["label"],
                "trigger_word": identity["trigger_word"],
                "prompt": prompt,
                "base_model": base_model,
                "job_id": job_id,
                "source_asset_ids": lora_item_id,
                "created_by": created_by,
            }
        )
        skipped_report = prov_skipped
        await cantemo.set_metadata(item_id, prov_fields, group_name=PROVENANCE_GROUP)
        try:
            await cantemo.create_relation(item_id, lora_item_id, relation_type=REL_GENERATED_WITH)
        except Exception as exc:
            _logger.error("[lora] relation %s -> %s failed: %s", item_id, lora_item_id, exc)
        created.append(item_id)

    filed = await _file_into(collection, created)

    return {
        "ok": True,
        "job_id": job_id,
        "items": created,
        "count": len(created),
        "filed": filed,
        "lora": {"item_id": lora_item_id, **identity},
        "provenance_skipped": skipped_report,
    }


def cleanup_workdir(workdir: str) -> None:
    """Training images are copies; the originals stay in the MAM."""
    shutil.rmtree(workdir, ignore_errors=True)


async def bootstrap_provenance_group() -> dict:
    """
    Report what would be needed to create the "AI Provenance" metadata group.

    Field creation is NOT on the v2 API surface -- /API/v2/metadata-schema/
    exposes reads plus choices/hierarchy writes, but new fields and groups go
    through Vidispine (/vs/metadataelement/) or the Portal admin UI. The account
    holds portal_manage_metadata_groups and portal_metadata_elements_create, so
    this is permitted; it just is not a v2 REST call. Left as a reported plan
    rather than a silent half-implementation.
    """
    existing = await cantemo.list_metadata_groups(limit=200)
    names = {g.get("name") for g in (existing or {}).get("results", [])}

    # Report which of our fields actually landed, so a half-finished UI session
    # is visible rather than showing up later as quietly missing metadata.
    present: set[str] = set()
    if PROVENANCE_GROUP in names:
        try:
            group = await cantemo.get_metadata_group(PROVENANCE_GROUP)
            present = {f.get("name") for f in (group or {}).get("fields", [])}
        except Exception as exc:
            _logger.error("[lora] could not read group %s: %s", PROVENANCE_GROUP, exc)

    return {
        "group": PROVENANCE_GROUP,
        "exists": PROVENANCE_GROUP in names,
        "create_via": "Portal admin UI (/vs/metadatamanagement/) or Vidispine metadata-field API",
        "missing": sorted(set(PROVENANCE_FIELD_IDS.values()) - present),
        "present": sorted(set(PROVENANCE_FIELD_IDS.values()) & present),
        # Every field is Text except the prompt, which is a Textarea. The code
        # writes strings throughout, so a Date/Timestamp/Multi-Choice field
        # would reject the value it is handed.
        "fields": [
            {"id": field_id, "ui_type": "Textarea" if field_id == "aiprov_prompt" else "Text"}
            for field_id in PROVENANCE_FIELD_IDS.values()
        ],
    }
