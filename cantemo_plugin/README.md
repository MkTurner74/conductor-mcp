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
Environment=CONDUCTOR_API_URL=https://web-production-c671d.up.railway.app
Environment=CONDUCTOR_API_TOKEN=<the real bearer token, never a placeholder>
```

Then:

```bash
systemctl daemon-reload
systemctl restart portal-web.service
```

Confirm it loaded: `/create_lora/` should return `302` (redirect to login)
or `200`, never `404` — a `404` means the app didn't register.

## Status (2026-09-03)

Installed and confirmed loading clean on Codemill's disposable playground
box (`13.60.16.167`, deleted ~October). Outbound network access from that
box to conductor-mcp confirmed working. Not yet fired for real — the
systemd override there currently holds a placeholder token, not the real
one. Once proven, hand off to Codemill to install on their production demo
Portal (`cantemo6.codemill.se`) — that box, not this one, is where the
actual IBC demo runs.
