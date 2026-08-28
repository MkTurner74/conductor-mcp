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

**Still unverified:** whether `python` (or `accelerate`) is on PATH once
$ENV_FILE is sourced. DEFAULT_LAUNCHER assumes plain `python`. Run the env probe
(run_discovery.py --probe-env) before the first paid training run.
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

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
            got = await cantemo.download_to(item_id, dest, shape_id=original.get("id"))
            if got:
                files.append(got)
            else:
                skipped.append({"item_id": item_id, "reason": "download failed"})
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

TRAIN_COMMAND_TEMPLATE = (
    'bash -lc \'source "$ENV_FILE" && {launcher} "$KOHYA_PATH/sdxl_train_network.py"'
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
    " --cache_latents'"
)

# Probes what the package environment actually provides once sourced -- the one
# thing discovery 00005 could not answer, because it never sourced $ENV_FILE.
ENV_PROBE_COMMAND = (
    "bash -lc '"
    "echo ===BEFORE===; echo PATH=$PATH; "
    'source "$ENV_FILE"; '
    "echo ===AFTER===; echo PATH=$PATH; "
    "echo ===WHICH===; "
    "for b in python python3 accelerate torchrun; do "
    'printf "%s -> " "$b"; command -v "$b" || echo MISSING; done; '
    "echo ===VERSION===; python -V 2>&1 || python3 -V 2>&1; "
    "echo ===TORCH===; "
    "python -c \"import torch,sys;print(torch.__version__, torch.cuda.is_available())\" 2>&1 | head -3; "
    "echo ===ENVFILE===; head -40 \"$ENV_FILE\"; "
    "echo ===DONE==='"
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
INFER_COMMAND_TEMPLATE = (
    'bash -lc \'source "$ENV_FILE" && {launcher} "$KOHYA_PATH/sdxl_gen_img.py"'
    ' --ckpt "$MODEL_BASE_HOME"'
    " --network_module networks.lora"
    " --network_weights {lora_path}"
    " --outdir {output_path}"
    ' --prompt "{prompt}"'
    " --images_per_prompt {count}"
    " --W {width} --H {height}"
    " --steps {steps}"
    " --seed {seed}'"
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

    # Quotes in a prompt would break out of the shell command; strip rather than
    # attempt to escape, since a demo prompt never needs them.
    safe_prompt = prompt.replace('"', "").replace("'", "").replace("\n", " ").strip()

    command = INFER_COMMAND_TEMPLATE.format(
        launcher=launcher,
        lora_path=node_lora_path or node_path(staged["path"]),
        output_path=output_path,
        prompt=safe_prompt,
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
}


def _provenance_fields(values: dict) -> list[dict]:
    """
    Flatten a provenance dict into Cantemo metadata fields, translating our
    logical names into the Portal's field ids.

    An unmapped key would be written under a name the Portal does not know and
    silently discarded, so it raises instead -- a provenance record that loses
    fields without saying so is worse than one that fails loudly.
    """
    out = []
    for key, value in values.items():
        if value is None:
            continue
        field_id = PROVENANCE_FIELD_IDS.get(key)
        if not field_id:
            raise KeyError(f"No Cantemo field id mapped for provenance key {key!r}")
        out.append({"name": field_id, "value": str(value)})
    return out


async def ingest_lora_to_cantemo(
    job_id: str,
    label: str,
    source_item_ids: list[str],
    base_model: str,
    trigger_word: str,
    created_by: str,
    notranscode: bool = True,
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
        if str(f.get("name") or f.get("path") or "").endswith(".safetensors")
    ]
    if not weights:
        return {"ok": False, "error": f"No .safetensors in outputs of job {job_id}", "outputs": outputs}

    weight = weights[0]
    url = weight.get("url") or weight.get("signed_url")
    if not url:
        return {"ok": False, "error": "Output file carried no signed URL", "file": weight}

    item = await cantemo.create_placeholder(title=f"LoRA -- {label}")
    item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
    if not item_id:
        return {"ok": False, "error": "Could not read new item id", "response": item}

    await cantemo.import_uri(item_id, url, notranscode=notranscode)
    await cantemo.set_metadata(
        item_id,
        _provenance_fields(
            {
                "provenance_kind": "lora",
                "label": label,
                "base_model": base_model,
                "trigger_word": trigger_word,
                "job_id": job_id,
                "source_asset_ids": ",".join(source_item_ids),
            }
        ),
        group_name=PROVENANCE_GROUP,
    )

    # The edges are the point: every training asset becomes a traversable
    # relation, so the MAM itself can answer "what was this trained on".
    linked, failed = [], []
    for src in source_item_ids:
        try:
            await cantemo.create_relation(item_id, src, relation_type=REL_TRAINED_FROM)
            linked.append(src)
        except Exception as exc:
            failed.append({"item_id": src, "error": str(exc)})

    return {
        "ok": True,
        "item_id": item_id,
        "job_id": job_id,
        "weights_file": weight.get("name") or weight.get("path"),
        "related": linked,
        "relation_failures": failed,
    }


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

    await cantemo.set_metadata(
        item_id,
        _provenance_fields(
            {
                "provenance_kind": "lora",
                "status": STATUS_SUBMITTED,
                "label": label,
                "base_model": base_model,
                "trigger_word": trigger_word,
                "job_id": job_id,
                "source_asset_ids": ",".join(source_item_ids),
            }
        ),
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
        item_id, _provenance_fields({"status": mapped}), group_name=PROVENANCE_GROUP
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
        if str(f.get("name") or f.get("path") or "").endswith(".safetensors")
    ]
    if not weights:
        await cantemo.set_metadata(
            item_id, _provenance_fields({"status": STATUS_FAILED}), group_name=PROVENANCE_GROUP
        )
        return {"ok": False, "error": f"No .safetensors in outputs of job {job_id}"}

    url = weights[0].get("url") or weights[0].get("signed_url")
    if not url:
        return {"ok": False, "error": "Output file carried no signed URL"}

    await cantemo.import_uri(item_id, url, notranscode=True)
    await cantemo.set_metadata(
        item_id, _provenance_fields({"status": STATUS_READY}), group_name=PROVENANCE_GROUP
    )
    return {"ok": True, "item_id": item_id, "weights_file": weights[0].get("name") or weights[0].get("path")}


async def ingest_generated_images(
    job_id: str,
    lora_item_id: str,
    prompt: str,
    base_model: str,
    created_by: str,
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
        if str(f.get("name") or f.get("path") or "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not images:
        return {"ok": False, "error": f"No images in outputs of job {job_id}"}

    created = []
    for idx, img in enumerate(images, start=1):
        url = img.get("url") or img.get("signed_url")
        if not url:
            continue
        item = await cantemo.create_placeholder(title=f"{prompt[:60]} ({idx})")
        item_id = item.get("id") or item.get("item_id") or (item.get("object") or {}).get("id")
        if not item_id:
            continue
        await cantemo.import_uri(item_id, url, notranscode=False)
        await cantemo.set_metadata(
            item_id,
            _provenance_fields(
                {
                    "provenance_kind": "generated_image",
                    "prompt": prompt,
                    "base_model": base_model,
                        "job_id": job_id,
                        }
            ),
            group_name=PROVENANCE_GROUP,
        )
        try:
            await cantemo.create_relation(item_id, lora_item_id, relation_type=REL_GENERATED_WITH)
        except Exception as exc:
            _logger.error("[lora] relation %s -> %s failed: %s", item_id, lora_item_id, exc)
        created.append(item_id)

    return {"ok": True, "job_id": job_id, "items": created, "count": len(created)}


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
