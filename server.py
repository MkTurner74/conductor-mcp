"""
Conductor MCP Server
Exposes Conductor by CoreWeave job management as MCP tools.
Any MCP-compatible AI agent can submit and manage render jobs via this server.
"""

import json
import os

from mcp.server.fastmcp import FastMCP

import conductor_client as conductor

mcp = FastMCP("Conductor")


@mcp.tool()
async def list_instance_types() -> str:
    """
    List all available hardware instance types on Conductor.
    Returns machine names, CPU/GPU specs, and indicative cost tiers.
    Use this before submitting a job to pick the right instance_type.
    """
    data = await conductor.list_instance_types()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_projects() -> str:
    """
    List all Conductor projects for the authenticated account.
    Use this to find valid project names before submitting a job.
    """
    data = await conductor.list_projects()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_software_packages() -> str:
    """
    List all available software packages on Conductor — Maya, Blender, Houdini, Nuke, Cinema 4D, etc.
    Returns package IDs and version info. Use package IDs when submitting a job.
    """
    data = await conductor.list_software_packages()
    return json.dumps(data, indent=2)


@mcp.tool()
async def list_jobs(job_id_start: int = None, job_id_end: int = None) -> str:
    """
    List render jobs on the account. Optionally filter by job ID range.

    Args:
        job_id_start: Start of job ID range to filter (optional)
        job_id_end:   End of job ID range to filter (optional)

    Returns status, progress, cost, and metadata for each job.
    """
    data = await conductor.list_jobs(job_id_start, job_id_end)
    return json.dumps(data, indent=2)


@mcp.tool()
async def submit_render_job(
    job_title: str,
    project: str,
    instance_type: str,
    software_package_ids: list[str],
    tasks: list[dict],
    output_path: str,
    priority: int = 5,
    preemptible: bool = True,
    notify: list[str] = None,
    metadata: dict = None,
) -> str:
    """
    Submit a render job to Conductor.

    Args:
        job_title:             Display name shown in the Conductor dashboard
        project:               Project name — use list_projects() to find valid names
        instance_type:         Machine type e.g. 'standard', 'highcpu', 'highmem'
                               Use list_instance_types() to see all options
        software_package_ids:  List of software package IDs to load
                               Use list_software_packages() to find IDs
        tasks:                 List of task dicts. Each task must have:
                                 - "command": the render command to run (str)
                                 - "frames": frame range e.g. "1-100" (str, optional)
                               Example: [{"command": "vray -sceneFile=/path/scene.vrscene", "frames": "1-50"}]
        output_path:           Where rendered output should be written
        priority:              Job priority 1–10. Higher = runs sooner. Default 5.
        preemptible:           Use preemptible/spot instances to reduce cost. Default True.
        notify:                Email addresses to notify on completion (optional)
        metadata:              Custom key-value pairs for reporting/tracking (optional)

    Returns job ID and submission status.
    Note: This tool assumes scene files are already uploaded to Conductor storage.
    For local file upload, use the Conductor desktop client or CLI before submitting.
    """
    payload: dict = {
        "job_title": job_title,
        "project": project,
        "machine_flavor": instance_type,
        "tasks_data": tasks,
        "output_path": output_path,
        "priority": priority,
        "preemptible": preemptible,
        "software_package_ids": software_package_ids,
    }
    if notify:
        payload["notify"] = notify
    if metadata:
        payload["metadata"] = metadata

    data = await conductor.submit_job(payload)
    return json.dumps(data, indent=2)


@mcp.tool()
async def kill_jobs(job_ids: list[int], action: str = "kill") -> str:
    """
    Cancel or hold one or more render jobs.

    Args:
        job_ids: List of integer job IDs to act on
        action:  What to do — 'kill' to cancel, 'hold' to pause (default: 'kill')

    Returns updated status for each affected job.
    """
    data = await conductor.kill_jobs(job_ids, action)
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_task_log(job_id: str, task_id: str) -> str:
    """
    Retrieve the log output for a specific task within a render job.
    Useful for diagnosing errors on failed tasks.

    Args:
        job_id:  The Conductor job ID (from list_jobs or submit_render_job response)
        task_id: The task ID within the job
    """
    data = await conductor.get_task_log(job_id, task_id)
    return json.dumps(data, indent=2) if isinstance(data, dict) else str(data)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
