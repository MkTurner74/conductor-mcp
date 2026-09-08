# Cantemo Portal plugin — Create LoRA

Adapted from Cantemo's own reference implementation,
[SearchPageExportExample](https://github.com/Cantemo/SearchPageExportExample)
— "an action based on user selection on the search page," pointed at LoRA
training instead of CSV export. Full design context:
`projects/coreweave-ibc-lora-demo/README.md` in the docs vault.

## What it does

Adds "Create LoRA" to the search page's multi-select gear menu. On click,
opens a small dialog (name + trigger word), then POSTs the selection and
form fields to this plugin's own Django endpoint, which calls this repo's
`button_api.py` REST routes (`POST /lora/train`) server-to-server — the
bearer token never reaches the browser.

## Install

```bash
cp -r create_lora /opt/cantemo/portal/portal/plugins/create_lora
chown -R www-data:www-data /opt/cantemo/portal/portal/plugins/create_lora
```

Set the environment on the `portal-web.service` unit (a systemd drop-in,
e.g. `/etc/systemd/system/portal-web.service.d/create_lora.conf`):

```ini
[Service]
Environment=CONDUCTOR_API_URL=https://conductor-mcp-production.up.railway.app
Environment=CONDUCTOR_API_TOKEN=<the real bearer token, never a placeholder>
```

Then:

```bash
systemctl daemon-reload
systemctl restart portal-web.service
```

Confirm it loaded: `/create_lora/` should return `302` (redirect to login)
or `200`, never `404` — a `404` means the app didn't register.

## Status (2026-09-07)

**Mechanism fully proven on Codemill's disposable playground box**
(`13.60.16.167`, deleted ~October): plugin installed, loads clean, real
bearer token set (was a placeholder as of 09-03 — fixed), unauthenticated
calls to conductor-mcp correctly get `401`, authenticated ones get past
auth cleanly. The button fires, reaches Conductor, and gets a real
structured response back — nothing left to prove about the wiring itself.

**A real training run only succeeds against `cantemo6.codemill.se`
content**, and this is expected, not a bug: conductor-mcp's Cantemo
connection is single-tenant, hardcoded at the service level to
`cantemo6.codemill.se` (the same instance Samsyn's own workflows depend
on) — it has no notion of "which Cantemo" per request. The playground
box's own local Cantemo Portal is a separate database with its own
`VX-NNNN` item ids that happen to overlap numerically with production but
refer to different assets — training against playground-selected items
predictably 404s/403s once it reaches Conductor, since those ids don't
exist (or aren't accessible) on `cantemo6.codemill.se`. Confirmed live,
2026-09-07: selecting playground F1-car test images and clicking Create
LoRA returned `RuntimeError: No trainable images among the selected
items`, citing exactly the production URL pattern
(`cantemo6.codemill.se/API/v2/items/<id>/formats/`) with 404/403 per item.

**This is why the next step has to be installing on `cantemo6.codemill.se`
itself** — not more playground iteration. The mechanism is proven; the
only thing left to validate is real content, which only exists there.
We don't have filesystem/SSH access to that box (Codemill kept it closed
and gave us the playground instead) — the install above needs to be run
by Codemill, or by us with temporary access if they're willing to grant
it now that the plugin is proven safe to install (same install steps,
same file, nothing playground-specific in the code — the only per-box
config is the systemd environment override).
