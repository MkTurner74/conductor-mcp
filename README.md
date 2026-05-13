# Conductor MCP Server

An MCP (Model Context Protocol) server for [Conductor by CoreWeave](https://conductortech.com).
Lets any MCP-compatible AI agent — Claude, Cursor, LangChain, AutoGen — submit and manage render jobs without custom integration code.

Built by [Entertainment Technologists Inc.](https://entertainmentconsultancy.com)

---

## What it does

Exposes 7 tools over MCP:

| Tool | What an agent can do |
|------|----------------------|
| `list_instance_types` | Query available hardware and cost tiers |
| `list_projects` | List Conductor projects on the account |
| `list_software_packages` | List available software (Maya, Blender, Houdini, Nuke, etc.) |
| `list_jobs` | List and filter render jobs by status and ID |
| `submit_render_job` | Submit a new render job |
| `kill_jobs` | Cancel or hold one or more jobs |
| `get_task_log` | Pull logs for a specific task |

---

## Quick start (5 minutes)

### 1. Get your Conductor API key
Go to [dashboard.conductortech.com/profile](https://dashboard.conductortech.com/profile) → **Get API Key**.
Download the key file and note its contents.

### 2. Run locally

```bash
git clone https://github.com/YOUR_ORG/conductor-mcp
cd conductor-mcp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and paste your API key
python server.py
```

The server starts on `http://localhost:8000`. The MCP endpoint is at `/sse`.

### 3. Connect to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "conductor": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

### 4. Connect to Cursor

Add to Cursor MCP settings:
```
URL: http://localhost:8000/sse
```

---

## Hosted instance (demo)

A reference instance is available at the URL provided by CoreWeave/ETI for evaluation.
You will need to supply your own Conductor API key — the hosted server never stores credentials.

To connect to the hosted instance, replace `http://localhost:8000` with the provided URL.

---

## Deploy your own (Railway)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Fork this repo
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set environment variables in Railway dashboard:
   - `CONDUCTOR_API_KEY` — your API key value
   - `CONDUCTOR_API_URL` — `https://api.conductortech.com` (default, usually no change needed)
4. Deploy — Railway assigns a public URL automatically

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONDUCTOR_API_KEY` | Yes | — | API key from your Conductor profile |
| `CONDUCTOR_API_URL` | No | `https://api.conductortech.com` | Conductor API base URL |
| `PORT` | No | `8000` | Port to listen on (Railway sets this automatically) |

---

## Example agent prompts

Once connected, you can ask an AI agent:

> "What render hardware is available on Conductor right now?"

> "Submit a Blender render job for the file at /projects/shot_005/scene.blend, frames 1 to 100, standard instance, output to /renders/shot_005/"

> "Show me all my active Conductor jobs."

> "Kill job 00695."

> "What went wrong with task 3 on job 00712?"

---

## Architecture

```
AI Agent (Claude, Cursor, LangChain, etc.)
        │  MCP / SSE
        ▼
  Conductor MCP Server  ◄── CONDUCTOR_API_KEY env var
        │  REST / HTTPS
        ▼
  Conductor API (conductortech.com)
        │
        ▼
  CoreWeave GPU infrastructure
```

Auth flow: API key is exchanged for a short-lived bearer token on first use. The token is cached in memory for the lifetime of the server process.

---

## Notes

- **File uploads**: This server handles job lifecycle. Scene file uploads are handled separately via the Conductor desktop client or CLI — run these before submitting a job via MCP.
- **Rover / cost arbitrage**: Multi-cloud cost optimisation is not yet in Conductor's public API. Will be added when available.
- **LoRA / AI training**: Conductor AI is in beta. Will be added once the API is stable.

---

## License

MIT — see [LICENSE](LICENSE)
