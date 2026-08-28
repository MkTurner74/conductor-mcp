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

## Status — 2026-08-28: LIVE

**https://conductor-mcp-production.up.railway.app** — deployed, with ciocore.

- Project `conductor-mcp` / service `conductor-mcp` / env `production`
- Bearer gate verified: 401 without a token, 200 with
- `tools/list` returns **18**, including all four LoRA tools
- A dry-run `submit_lora_training` through the hosted server pulled 3 images
  from Cantemo, resolved both kohya packages and built the job
- Samsyn repointed (`CONDUCTOR_MCP_URL` + `CONDUCTOR_MCP_TOKEN`) and redeployed

### The build failure, and what it actually was

Three deploys failed before the dashboard log gave up the real error:

```
The user requested PyJWT==2.9.0
mcp 1.27.1 depends on pyjwt>=2.10.1
ERROR: ResolutionImpossible
```

ciocore 9.1.1 **and** 9.2.0 hard-pin `pyjwt==2.9.0`; mcp declares `>=2.10.1`.
pip refuses the pair. The conflict is **declared, not real** — the local venv
that submitted Conductor job 00005 runs mcp 1.27.1 against pyjwt 2.9.0. The
Dockerfile reproduces that proven combination rather than bending either side,
and asserts it at build time so a future bump fails the build, not the first
submission.

**Railway builds with railpack now**, which ignores `railway.toml` AND
`nixpacks.toml` — the `pip install ciocore` phase nixpacks.toml existed to add
never ran. Railpack defers to a Dockerfile, which is why one exists. Both older
config files are now effectively dead; leave or delete them, they do nothing.

`requirements.txt` stays lean and the Dockerfile is `.vercelignore`d, so the
Vercel deployment is untouched and still serves the read-only tools.

### Account note

The Railway project sits under **mark@reallyme.me** (personal). Since it fronts
Samsyn, the standing rule says ETI-owned — worth moving before it becomes
load-bearing.

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
