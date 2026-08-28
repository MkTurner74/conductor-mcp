# Deploying conductor-mcp to Railway

## Why Railway and not Vercel

The Vercel deployment (`conductor-mcp-one.vercel.app`) is deliberately **lean**:
`ciocore` is kept out of `requirements.txt` so the serverless bundle stays
small. That instance can therefore **read** — `list_jobs`, `list_software_packages`,
`get_job_outputs` — but it **cannot submit a job**, and it cannot upload a file.

The LoRA workflow needs both:

- **Submission** does not strictly need ciocore. Probed 2026-08-27:
  `POST /api/v1/jobs` answers **422 "Job is missing required attributes: project"**,
  not 404 — the endpoint exists and validates. The "may 404" note in
  `conductor_client.submit_job` is stale.
- **Uploading the training images does.** CoreWeave render nodes have no
  egress, so the dataset has to be pushed into Conductor storage before the job
  runs. That upload is ciocore's worker pool (md5 + chunking). There is no tidy
  REST equivalent — reimplementing it would be rebuilding the SDK.

So: Railway, where `nixpacks.toml` chains a `pip install ciocore==9.1.1` phase
after Nixpacks' own install.

## Status — 2026-08-28

**The service is LIVE and serving all 18 tools**, but **without ciocore**, so it can
read and it can build a job, but it cannot upload a dataset — which means LoRA
training still cannot run.

- Project `conductor-mcp` (workspace "Mark Turner's Projects"), service
  `conductor-mcp`, environment `production`
- URL: **https://conductor-mcp-production.up.railway.app**
- Bearer gate confirmed: 401 without a token, 200 with
- `tools/list` returns 18, including `submit_lora_training`, `ingest_lora_to_mam`,
  `submit_lora_inference`, `ingest_generated_to_mam`
- All 8 environment variables set

**Open problem: adding ciocore fails the build.** Three attempts, all "Deploy
failed", and the failure happens during railpack's *prepare* phase — before pip
produces any output. `railway logs --build` unhelpfully replays the last
*successful* build, so the real error is only visible in the dashboard:

> https://railway.com/project/3a4020c7-8047-4d13-b9fe-b5a9b48f65f5/service/243af1f4-d779-41a1-933d-c3ecf8ae8a20

Ruled out so far:
- **Python version.** ciocore 9.1.1 ships a universal `py2.py3-none-any` wheel with
  no `requires_python`, so railpack's Python 3.13 is not the issue.
- **The PyJWT conflict.** ciocore hard-pins `pyjwt==2.9.0` while our `PyJWT>=2.8.0`
  had resolved to 2.13.0. requirements.txt now pins 2.9.0 to match — correct
  regardless, but it did not fix the build.
- **A malformed requirements.txt.** Simplifying the file made no difference.

Next thing to try: read the dashboard build log for the actual error. If it turns
out to be an image-size or timeout limit, the fallback is a `Dockerfile` (railpack
defers to one when present), which also ends the nixpacks/railpack config drift.

**Note:** `railway.toml` declares `builder = "nixpacks"` but Railway now uses
**railpack**, which ignores both `railway.toml` and `nixpacks.toml`. The
`pip install ciocore` phase that `nixpacks.toml` was written to add never ran —
that is why ciocore had to move into `requirements.txt` at all.

**Samsyn is still pointed at the Vercel instance.** Nothing is broken; the
repoint below is deliberately not done until ciocore is working.

## One-time setup

```bash
cd C:\Users\mktur\code\conductor-mcp

npx @railway/cli login          # opens a browser
npx @railway/cli init           # create the project (name it conductor-mcp)
```

## Environment variables

Set these on the Railway service before the first deploy.

| Variable | Value | Why |
|---|---|---|
| `MCP_TRANSPORT` | `sse` | **Required.** `server.py` defaults to stdio; without this it never opens a port and Railway's health check fails. |
| `CONDUCTOR_API_KEY` | contents of `conductor_key.json`, as one JSON string | ciocore reads `CONDUCTOR_API_KEY` or `CONDUCTOR_API_KEY_PATH` — not our `_FILE` variant. |
| `MCP_AUTH_TOKEN` | same value as Samsyn's `CONDUCTOR_MCP_TOKEN` | Bearer gate. This server fronts an account that spends real money — it must never run open. |
| `CANTEMO_URL` | `https://cantemo6.codemill.se` | |
| `CANTEMO_API_TOKEN` | the Portal `auth-token` | |
| `CANTEMO_USER` / `CANTEMO_PASSWORD` | Portal login | Only needed for Vidispine-level calls. |
| `STATELESS_HTTP` | `1` | Matches the Vercel entry's posture; harmless on a persistent host. |

Retrieve the existing bearer token rather than inventing a new one — otherwise
Samsyn's `/api/conductor` route stops authenticating:

```bash
cd ..\owg-core\app
npx vercel env pull .env.vercel      # CONDUCTOR_MCP_TOKEN is in there
```

## Deploy

```bash
cd C:\Users\mktur\code\conductor-mcp
npx @railway/cli up
```

`railway up` ships the **working tree**, so make sure you are on
`feat/cantemo-lora-integration` — master does not have the Cantemo or LoRA
tools.

## Point Samsyn at it

```bash
cd ..\owg-core\app
npx vercel env rm CONDUCTOR_MCP_URL production --yes
npx vercel env add CONDUCTOR_MCP_URL production     # paste the Railway URL
npm run deploy:prod
```

## Verify

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://<railway-url>/          # 401 without a token = gate working
curl -s -H "Authorization: Bearer <MCP_AUTH_TOKEN>" \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  https://<railway-url>/ | head -c 400
```

Expect **18 tools**, including `submit_lora_training`, `ingest_lora_to_mam`,
`submit_lora_inference`, `ingest_generated_to_mam` and `cantemo_search_assets`.
If you see 10, the deploy is running master and needs the branch.

## Gotcha worth remembering

Windows `openssl` emits CRLF. A trailing `\r` in a bearer header makes an edge
proxy return 400 before the function is even reached — this cost time on the
Vercel deployment. Generate tokens with `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`.
