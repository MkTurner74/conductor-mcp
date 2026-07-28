"""
Samsyn × Conductor render→video PROOF pipeline.

One Conductor job that, on a CoreWeave node, with NO uploaded files:
  1. builds a scene at runtime via hython (extruded frame-number text — a simple
     animating 3D object),
  2. renders it with Karma CPU to a **PNG** sequence, then
  3. **assembles the frames into an MP4 with FFmpeg in the same task**.

The MP4 lands in the job output path; get_job_outputs() returns a signed URL,
which a Botverse transcode step (or Frame.io upload) then consumes.

Two things it deliberately demonstrates:
  • ciocore SDK submission (conductor_submit.Submit) — the PROVEN path the
    Conductor team uses, not our older raw-REST /api/v1/jobs POST.
  • FFmpeg on a Conductor/CoreWeave node — the mechanism that could move
    Botverse's encoding off AWS Fargate onto CoreWeave. See
    docs: projects/owg-samsyn/conductor-ffmpeg-render-pipeline.md.

Adapted from the Conductor team's houdini21_karma_text_submit.py (proven to
submit). Changes: EXR→PNG output, added the FFmpeg assemble step, demo-sized
defaults (fewer frames, 720p) to keep the test render cheap.

Cost note: this submits a REAL paid render on CoreWeave. Keep frames/res small.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import logging
import pprint
import textwrap

import ciocore.conductor_submit
import ciocore.loggeria
from ciocore import data as coredata
from ciocore.package_environment import PackageEnvironment

logger = logging.getLogger(f"{ciocore.loggeria.CONDUCTOR_LOGGER_NAME}.render_and_assemble_demo")

# Demo-sized defaults — a real render costs money, so keep it small.
OUTPUT_PATH = "/my_renders/samsyn_render_demo"
PROJECT = "TestProject"   # the one project on this account (list_projects); NOT "default"
INSTANCE_TYPE = "cw-xeonv3-32"
FRAME_START = 1
FRAME_END = 24            # 1 second at 24fps — enough to prove motion, cheap
RESOLUTION = (1280, 720)  # 720p — cheaper/faster than 1080p for a proof
FPS = 24

# Env var names for the runtime render script (script-generated, not shell).
E_OUT = "CONDUCTOR_HOUDINI_RENDER_OUTPUT"
E_FS = "CONDUCTOR_HOUDINI_FRAME_START"
E_FE = "CONDUCTOR_HOUDINI_FRAME_END"
E_RX = "CONDUCTOR_HOUDINI_RES_X"
E_RY = "CONDUCTOR_HOUDINI_RES_Y"
E_SCRIPT = "CONDUCTOR_HOUDINI_PY_SCRIPT"

# ── The runtime hython scene: extruded text of the current frame, Karma CPU,
#    rendered to a PNG sequence (8-bit; Karma applies sRGB for 8-bit formats,
#    so no manual tonemap needed — which is why PNG keeps FFmpeg assembly
#    trivial vs. linear EXR). Mirrors the Conductor team's proven scene. ──
HOUDINI_SCRIPT = textwrap.dedent(
    r"""
    import os
    import hou

    def set_parm(node, names, value, required=True):
        for name in names:
            parm = node.parmTuple(name) if isinstance(value, (tuple, list)) else node.parm(name)
            if parm is not None:
                parm.set(value)
                return name
        if required:
            raise hou.OperationFailed(f"Could not find any of {names!r} on {node.path()}")
        return None

    output_path = os.environ["CONDUCTOR_HOUDINI_RENDER_OUTPUT"]
    frame_start = int(os.environ.get("CONDUCTOR_HOUDINI_FRAME_START", "1"))
    frame_end = int(os.environ.get("CONDUCTOR_HOUDINI_FRAME_END", "24"))
    res_x = int(os.environ.get("CONDUCTOR_HOUDINI_RES_X", "1280"))
    res_y = int(os.environ.get("CONDUCTOR_HOUDINI_RES_Y", "720"))

    os.makedirs(output_path, exist_ok=True)
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.setFrame(frame_start)
    hou.playbar.setFrameRange(frame_start, frame_end)
    hou.playbar.setPlaybackRange(frame_start, frame_end)

    obj = hou.node("/obj")
    geo = obj.createNode("geo", "text_geo")
    for child in geo.children():
        child.destroy()

    font = geo.createNode("font", "frame_text")
    font.parm("text").setExpression("$F", hou.exprLanguage.Hscript)
    set_parm(font, ("fontsize",), 2.0)
    set_parm(font, ("halign", "alignx"), 1, required=False)
    set_parm(font, ("valign", "aligny"), 1, required=False)

    extrude = geo.createNode("polyextrude", "extrude")
    extrude.setInput(0, font)
    set_parm(extrude, ("distance", "dist"), 0.35)
    # Gentle spin so the assembled video obviously moves.
    extrude_out = extrude

    xform = geo.createNode("xform", "spin")
    xform.setInput(0, extrude_out)
    xform.parm("ry").setExpression("$F*6", hou.exprLanguage.Hscript)

    normal = geo.createNode("normal", "normals")
    normal.setInput(0, xform)

    mat_net = hou.node("/mat") or hou.node("/").createNode("matnet", "mat")
    principled = mat_net.createNode("principledshader::2.0", "text_material")
    set_parm(principled, ("basecolorr",), 0.1)
    set_parm(principled, ("basecolorg",), 0.45)
    set_parm(principled, ("basecolorb",), 1.0)
    set_parm(principled, ("rough", "roughness"), 0.25, required=False)

    material = geo.createNode("material", "assign_material")
    material.setInput(0, normal)
    set_parm(material, ("shop_materialpath1", "shop_materialpath"), principled.path())
    material.setDisplayFlag(True)
    material.setRenderFlag(True)

    cam = obj.createNode("cam", "render_cam")
    set_parm(cam, ("t",), (0.0, 0.5, 9.0))
    set_parm(cam, ("r",), (-5.0, 0.0, 0.0))
    set_parm(cam, ("resx",), res_x, required=False)
    set_parm(cam, ("resy",), res_y, required=False)

    key = obj.createNode("hlight", "key_light")
    set_parm(key, ("t",), (4.0, 8.0, 6.0))
    set_parm(key, ("light_intensity", "intensity"), 2.5, required=False)
    fill = obj.createNode("hlight", "fill_light")
    set_parm(fill, ("t",), (-5.0, 3.0, 4.0))
    set_parm(fill, ("light_intensity", "intensity"), 1.0, required=False)

    stage = hou.node("/stage") or hou.node("/").createNode("stage", "stage")
    scene_import = stage.createNode("sceneimport", "import_obj")
    set_parm(scene_import, ("objects", "objpattern", "objectpattern"), "*")
    set_parm(scene_import, ("importobjects", "importobj"), 1, required=False)

    def find_camera(lop_node, preferred="render_cam"):
        from pxr import UsdGeom
        found, preferred_hits = [], []
        for prim in lop_node.stage().Traverse():
            if prim.IsA(UsdGeom.Camera):
                p = prim.GetPath().pathString
                found.append(p)
                if preferred in p:
                    preferred_hits.append(p)
        if preferred_hits:
            return preferred_hits[0]
        if found:
            return found[0]
        raise hou.OperationFailed("No USD camera prims found after Scene Import.")

    karma = stage.createNode("karma", "karma_render")
    karma.setInput(0, scene_import)
    karma.setDisplayFlag(True)
    camera_path = find_camera(karma)
    set_parm(karma, ("camera",), camera_path)
    # PNG (8-bit) output — Karma encodes sRGB for 8-bit formats, so FFmpeg can
    # assemble directly with no tonemap. frame.$F4.png -> frame.0001.png ...
    set_parm(karma, ("picture", "outputpicture", "vm_picture"), f"{output_path}/frame.$F4.png")
    set_parm(karma, ("resolution", "res"), (res_x, res_y))
    set_parm(karma, ("engine", "karma_engine"), "cpu")

    usdrender = hou.node("/out").createNode("usdrender", "karma_usdrender")
    set_parm(usdrender, ("loppath", "lopoutput", "loppath"), karma.path())
    set_parm(usdrender, ("overcamera", "override_camera", "overridecam", "camera"), camera_path, required=False)
    set_parm(usdrender, ("trange",), 1)
    set_parm(usdrender, ("f1",), frame_start)
    set_parm(usdrender, ("f2",), frame_end)
    set_parm(usdrender, ("f3",), 1)
    set_parm(usdrender, ("mkpath",), 1, required=False)

    print(f"Rendering frames {frame_start}-{frame_end} -> {output_path}/frame.####.png")
    usdrender.render(frame_range=(frame_start, frame_end), output_progress=True)
    print("Render complete.")
    """
).strip()

# ── FFmpeg resolution on the worker ──────────────────────────────────────────
# THE STRATEGIC BIT (see the doc): FFmpeg on a Conductor/CoreWeave node.
# For a Houdini job we can borrow Houdini's *bundled* ffmpeg ($HFS/bin/ffmpeg),
# which is guaranteed present with the package and needs no egress. For a job
# with no DCC package (e.g. a future Botverse-style encode-only job on
# CoreWeave), you SIDE-LOAD a static ffmpeg at runtime — that pattern is the
# SIDELOAD_FFMPEG_SNIPPET below, kept here on purpose per the off-AWS plan.
# FFmpeg resolution on the worker. EMPIRICAL FINDING (job 00001, 2026-07-28):
# Houdini 21 does NOT ship ffmpeg at $HFS/bin, and CoreWeave render nodes have
# no system ffmpeg — so the assemble step MUST side-load a static build. We
# still try $HFS/bin/ffmpeg and PATH first (cheap, no egress) and fall back to
# the side-load. This side-load is exactly the pattern that would let Botverse
# encode on CoreWeave (no DCC package to borrow ffmpeg from) — see the doc.
# Wrapped in { ...; } so `hython && { resolve; } && ffmpeg` chains correctly.
# NB: the side-load needs outbound egress from the node (johnvansickle static).
FFMPEG_RESOLVE = (
    '{ FF="$HFS/bin/ffmpeg"; '
    '[ -x "$FF" ] || FF="$(command -v ffmpeg)"; '
    '[ -x "$FF" ] || { '
    'echo "no ffmpeg on node - side-loading static build"; '
    'curl -Lso /tmp/ff.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz '
    '&& mkdir -p /tmp/ff && tar xJf /tmp/ff.tar.xz -C /tmp/ff --strip-components=1 '
    '&& FF=/tmp/ff/ffmpeg; }; '
    '[ -x "$FF" ] || { echo "ffmpeg unavailable (side-load failed - node egress?)"; exit 3; }; }'
)


def build_task_command(output_path: str, frame_start: int, fps: int) -> str:
    """hython render, then FFmpeg-assemble the PNG frames into an MP4, one task."""
    render = (
        'hython -c "import os,gzip,base64; '
        f"exec(gzip.decompress(base64.b64decode(os.environ['{E_SCRIPT}'])).decode())\""
    )
    assemble = (
        f'{FFMPEG_RESOLVE} && '
        f'"$FF" -y -framerate {fps} -start_number {frame_start} '
        f'-i "{output_path}/frame.%04d.png" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart '
        f'"{output_path}/render.mp4"'
    )
    return f"{render} && {assemble}"


def encode_script(script: str) -> str:
    return base64.b64encode(gzip.compress(script.encode("utf-8"))).decode("ascii")


def find_latest_houdini_21(tree_data):
    best_name, best_version = None, None
    for name in tree_data.to_path_list():
        if "/" in name or not name.endswith(" linux"):
            continue
        parts = name.split()
        if len(parts) == 3 and parts[0] == "houdini" and parts[1].startswith("21."):
            if best_version is None or parts[1] > best_version:
                best_name, best_version = name, parts[1]
    if not best_name:
        raise RuntimeError("No Houdini 21 linux package in the Conductor software tree.")
    return tree_data.find_by_name(best_name), best_version


def build_env(output_path, frame_start, frame_end, resolution):
    return {
        E_OUT: output_path,
        E_FS: str(frame_start),
        E_FE: str(frame_end),
        E_RX: str(resolution[0]),
        E_RY: str(resolution[1]),
        E_SCRIPT: encode_script(HOUDINI_SCRIPT),
    }


def submit(output_path, project, instance_type, frame_start, frame_end, resolution, fps, dry_run):
    coredata.init(product="houdini")
    tree = coredata.data()["software"]
    package, version = find_latest_houdini_21(tree)
    logger.info("Houdini package: %s (%s)", package.get("package_id"), version)

    env = PackageEnvironment()
    env.extend(package)
    env_vars = dict(env)
    env_vars.update(build_env(output_path, frame_start, frame_end, resolution))

    cmd = build_task_command(output_path, frame_start, fps)

    job_args = {
        "job_title": f"Samsyn render→video demo (Houdini {version}) {frame_start}-{frame_end} @ {fps}fps",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": [package["package_id"]],
        "upload_only": False,
        "force": False,
        "local_upload": False,
        "preemptible": True,           # a proof render — cheaper, retry on preempt
        "autoretry_policy": {"preempted": {"max_retries": 1}},
        "output_path": output_path,
        "environment": env_vars,
        "upload_paths": [],
        "scout_frames": str(frame_start),
        "tasks_data": [{"command": cmd, "frames": "1"}],
    }

    logger.info("Task command:\n%s", cmd)
    if dry_run:
        logger.warning("DRY RUN -- not submitting. job_args below:")
        logger.info("\n%s", pprint.pformat(job_args, indent=2))
        return {"dry_run": True}

    logger.info("Submitting to Conductor (this is a REAL paid render)…")
    result = ciocore.conductor_submit.Submit(job_args).main()
    logger.info("Submission result: %s", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Submit the Samsyn render→video proof job to Conductor.")
    p.add_argument("--output-path", default=OUTPUT_PATH)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument("--frame-start", type=int, default=FRAME_START)
    p.add_argument("--frame-end", type=int, default=FRAME_END)
    p.add_argument("--resolution", nargs=2, type=int, metavar=("W", "H"), default=RESOLUTION)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--dry-run", action="store_true", help="Print job_args and the task command; do NOT submit.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    ciocore.loggeria.setup_conductor_logging(logger_level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    submit(
        output_path=args.output_path,
        project=args.project,
        instance_type=args.instance_type,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        resolution=tuple(args.resolution),
        fps=args.fps,
        dry_run=args.dry_run,
    )
