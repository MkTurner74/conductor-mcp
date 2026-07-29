"""
SDK-based Conductor render submission for the Samsyn pipeline.

Submits an inline-scene Houdini/Karma render via the ciocore SDK (the PROVEN
path — the raw-REST /api/v1/jobs submit in conductor_client.py never worked).
No file uploads: the scene is built at runtime from an embedded hython script.

Renders a PNG frame sequence ONLY — assembly to video is a downstream step
(Botverse assemble_sequence, or client-side), because CoreWeave render nodes
have no egress and Houdini 21 ships no ffmpeg, so in-node assembly isn't
available (see projects/owg-samsyn/conductor-ffmpeg-render-pipeline.md).

Proven live: jobs 00001/00002 rendered 24 frames each on cw-xeonv3-32.
"""

from __future__ import annotations

import base64
import gzip
import os
import textwrap

# NB: ciocore (the Conductor SDK) is imported LAZILY inside submit_houdini_render,
# NOT at module top. It's a heavy dep that lives only in a full Python env (local
# venv / a persistent host), and is deliberately kept OUT of requirements.txt so
# the lean Vercel deployment (read-only tools + get_job_outputs, which Samsyn
# depends on) still imports and runs. submit_houdini_render therefore only works
# where ciocore is installed; the other tools work everywhere.

E_OUT = "CONDUCTOR_HOUDINI_RENDER_OUTPUT"
E_FS = "CONDUCTOR_HOUDINI_FRAME_START"
E_FE = "CONDUCTOR_HOUDINI_FRAME_END"
E_RX = "CONDUCTOR_HOUDINI_RES_X"
E_RY = "CONDUCTOR_HOUDINI_RES_Y"
E_SCRIPT = "CONDUCTOR_HOUDINI_PY_SCRIPT"

# Inline hython scene: a spinning extruded frame-number 3D object, Karma CPU,
# rendered to an 8-bit PNG sequence (Karma applies sRGB for 8-bit, so frames are
# display-ready and trivial to assemble). Proven in jobs 00001/00002.
HOUDINI_SCRIPT = textwrap.dedent(
    r"""
    import os, hou
    def set_parm(node, names, value, required=True):
        for name in names:
            parm = node.parmTuple(name) if isinstance(value, (tuple, list)) else node.parm(name)
            if parm is not None:
                parm.set(value); return name
        if required:
            raise hou.OperationFailed(f"missing {names!r} on {node.path()}")
        return None
    out = os.environ["CONDUCTOR_HOUDINI_RENDER_OUTPUT"]
    fs = int(os.environ.get("CONDUCTOR_HOUDINI_FRAME_START", "1"))
    fe = int(os.environ.get("CONDUCTOR_HOUDINI_FRAME_END", "24"))
    rx = int(os.environ.get("CONDUCTOR_HOUDINI_RES_X", "1280"))
    ry = int(os.environ.get("CONDUCTOR_HOUDINI_RES_Y", "720"))
    os.makedirs(out, exist_ok=True)
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.setFrame(fs); hou.playbar.setFrameRange(fs, fe); hou.playbar.setPlaybackRange(fs, fe)
    obj = hou.node("/obj")
    geo = obj.createNode("geo", "text_geo")
    for c in geo.children(): c.destroy()
    font = geo.createNode("font", "frame_text")
    font.parm("text").setExpression("$F", hou.exprLanguage.Hscript)
    set_parm(font, ("fontsize",), 2.0)
    set_parm(font, ("halign", "alignx"), 1, required=False)
    set_parm(font, ("valign", "aligny"), 1, required=False)
    extrude = geo.createNode("polyextrude", "extrude"); extrude.setInput(0, font)
    set_parm(extrude, ("distance", "dist"), 0.35)
    spin = geo.createNode("xform", "spin"); spin.setInput(0, extrude)
    spin.parm("ry").setExpression("$F*6", hou.exprLanguage.Hscript)
    nrm = geo.createNode("normal", "normals"); nrm.setInput(0, spin)
    matnet = hou.node("/mat") or hou.node("/").createNode("matnet", "mat")
    mat = matnet.createNode("principledshader::2.0", "text_material")
    set_parm(mat, ("basecolorr",), 0.1); set_parm(mat, ("basecolorg",), 0.45); set_parm(mat, ("basecolorb",), 1.0)
    set_parm(mat, ("rough", "roughness"), 0.25, required=False)
    asn = geo.createNode("material", "assign"); asn.setInput(0, nrm)
    set_parm(asn, ("shop_materialpath1", "shop_materialpath"), mat.path())
    asn.setDisplayFlag(True); asn.setRenderFlag(True)
    cam = obj.createNode("cam", "render_cam")
    set_parm(cam, ("t",), (0.0, 0.5, 9.0)); set_parm(cam, ("r",), (-5.0, 0.0, 0.0))
    set_parm(cam, ("resx",), rx, required=False); set_parm(cam, ("resy",), ry, required=False)
    k = obj.createNode("hlight", "key"); set_parm(k, ("t",), (4.0, 8.0, 6.0)); set_parm(k, ("light_intensity", "intensity"), 2.5, required=False)
    f = obj.createNode("hlight", "fill"); set_parm(f, ("t",), (-5.0, 3.0, 4.0)); set_parm(f, ("light_intensity", "intensity"), 1.0, required=False)
    stage = hou.node("/stage") or hou.node("/").createNode("stage", "stage")
    si = stage.createNode("sceneimport", "import_obj")
    set_parm(si, ("objects", "objpattern", "objectpattern"), "*")
    set_parm(si, ("importobjects", "importobj"), 1, required=False)
    def find_cam(lop, pref="render_cam"):
        from pxr import UsdGeom
        found, hits = [], []
        for p in lop.stage().Traverse():
            if p.IsA(UsdGeom.Camera):
                s = p.GetPath().pathString; found.append(s)
                if pref in s: hits.append(s)
        if hits: return hits[0]
        if found: return found[0]
        raise hou.OperationFailed("no USD camera after Scene Import")
    karma = stage.createNode("karma", "karma_render"); karma.setInput(0, si); karma.setDisplayFlag(True)
    cpath = find_cam(karma); set_parm(karma, ("camera",), cpath)
    set_parm(karma, ("picture", "outputpicture", "vm_picture"), f"{out}/frame.$F4.png")
    set_parm(karma, ("resolution", "res"), (rx, ry)); set_parm(karma, ("engine", "karma_engine"), "cpu")
    usd = hou.node("/out").createNode("usdrender", "karma_usdrender")
    set_parm(usd, ("loppath", "lopoutput", "loppath"), karma.path())
    set_parm(usd, ("overcamera", "override_camera", "overridecam", "camera"), cpath, required=False)
    set_parm(usd, ("trange",), 1); set_parm(usd, ("f1",), fs); set_parm(usd, ("f2",), fe); set_parm(usd, ("f3",), 1)
    set_parm(usd, ("mkpath",), 1, required=False)
    print(f"Rendering {fs}-{fe} -> {out}/frame.####.png")
    usd.render(frame_range=(fs, fe), output_progress=True)
    print("Render complete.")
    """
).strip()


def _ensure_ciocore_auth() -> None:
    """ciocore reads CONDUCTOR_API_KEY (json) or CONDUCTOR_API_KEY_PATH. Our own
    client uses CONDUCTOR_API_KEY / CONDUCTOR_API_KEY_FILE. Bridge them so the
    SDK authenticates in both hosted (json env) and local (key file) contexts."""
    if os.environ.get("CONDUCTOR_API_KEY") or os.environ.get("CONDUCTOR_API_KEY_PATH"):
        return
    path = os.environ.get("CONDUCTOR_API_KEY_FILE")
    if path and os.path.exists(path):
        os.environ["CONDUCTOR_API_KEY_PATH"] = path


def _encode(script: str) -> str:
    return base64.b64encode(gzip.compress(script.encode("utf-8"))).decode("ascii")


def _find_houdini21(tree):
    best_name = best_ver = None
    for name in tree.to_path_list():
        if "/" in name or not name.endswith(" linux"):
            continue
        parts = name.split()
        if len(parts) == 3 and parts[0] == "houdini" and parts[1].startswith("21."):
            if best_ver is None or parts[1] > best_ver:
                best_name, best_ver = name, parts[1]
    if not best_name:
        raise RuntimeError("No Houdini 21 linux package in the Conductor software tree.")
    return tree.find_by_name(best_name), best_ver


def submit_houdini_render(
    project: str = "TestProject",
    instance_type: str = "cw-xeonv3-32",
    output_path: str = "/my_renders/samsyn_render",
    frame_start: int = 1,
    frame_end: int = 24,
    res_x: int = 1280,
    res_y: int = 720,
    preemptible: bool = True,
    dry_run: bool = False,
) -> dict:
    """Submit the inline Houdini/Karma PNG-sequence render via the ciocore SDK.
    Returns {jid, job, output_path, frames} on success (or {dry_run, ...})."""
    try:
        import ciocore.conductor_submit
        from ciocore import data as coredata
        from ciocore.package_environment import PackageEnvironment
    except ImportError as e:
        raise RuntimeError(
            "ciocore (Conductor SDK) is not installed in this environment, so "
            "render submission is unavailable here. Run the MCP from a full "
            "Python env with ciocore (local venv or a persistent host), not the "
            "lean Vercel deployment."
        ) from e

    _ensure_ciocore_auth()
    coredata.init(product="houdini")
    package, version = _find_houdini21(coredata.data()["software"])

    env = PackageEnvironment()
    env.extend(package)
    env_vars = dict(env)
    env_vars.update({
        E_OUT: output_path, E_FS: str(frame_start), E_FE: str(frame_end),
        E_RX: str(res_x), E_RY: str(res_y), E_SCRIPT: _encode(HOUDINI_SCRIPT),
    })

    # Render-only task (no ffmpeg — assembly is a downstream step).
    cmd = (
        'hython -c "import os,gzip,base64; '
        f"exec(gzip.decompress(base64.b64decode(os.environ['{E_SCRIPT}'])).decode())\""
    )

    job_args = {
        "job_title": f"Samsyn Houdini render {frame_start}-{frame_end} ({res_x}x{res_y})",
        "project": project,
        "instance_type": instance_type,
        "software_package_ids": [package["package_id"]],
        "upload_only": False, "force": False, "local_upload": False,
        "preemptible": preemptible,
        "autoretry_policy": {"preempted": {"max_retries": 1}},
        "output_path": output_path,
        "environment": env_vars,
        "upload_paths": [],
        "scout_frames": str(frame_start),
        "tasks_data": [{"command": cmd, "frames": "1"}],
    }

    if dry_run:
        return {"dry_run": True, "houdini_version": version, "package_id": package["package_id"],
                "task_command": cmd, "output_path": output_path,
                "frames": f"{frame_start}-{frame_end}"}

    result, code = ciocore.conductor_submit.Submit(job_args).main()
    return {"jid": result.get("jid"), "job": result.get("job"), "code": code,
            "output_path": output_path, "frames": f"{frame_start}-{frame_end}",
            "status": result.get("status")}


async def render_status(jid: str) -> dict:
    """Status for a job by jid. NB: conductor_client.list_jobs range filter does
    NOT filter (returns job 1 regardless) — so we list all and match jid."""
    import conductor_client as cc
    jobs = await cc.list_jobs()
    data = jobs.get("data", jobs) if isinstance(jobs, dict) else jobs
    want = str(jid).zfill(5)
    for j in (data if isinstance(data, list) else [data]):
        if str(j.get("jid")) == want:
            return {k: j.get(k) for k in
                    ("jid", "status", "status_description", "running", "success",
                     "failed", "pending", "holding", "tasks", "total_runtime", "title")}
    return {"jid": want, "status": "not_found"}
