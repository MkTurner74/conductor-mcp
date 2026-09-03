"""
Views for the "Create LoRA" plugin.

GET  /create_lora/          -- returns the small form shown in the dialog
POST /create_lora/submit/   -- reads the user's selection + form fields,
                                calls conductor-mcp's REST button routes
                                (button_api.py in the conductor-mcp repo),
                                returns the result as JSON for the dialog's
                                success/error handler.

The outbound call is server-to-server (Portal -> conductor-mcp), never from
the browser -- the bearer token never reaches the client. Matches the
design already agreed with Codemill.
"""
import logging
import os

import requests
from django.http import JsonResponse
from rest_framework.response import Response

from portal.generic.baseviews import CView
from portal.generic.decorators import HasAnyRole
from portal.utils.general import get_item_ids_from_request

log = logging.getLogger(__name__)

# Set on the portal-web.service environment (systemd unit override) --
# never hardcoded, never committed.
CONDUCTOR_API_URL = os.environ.get("CONDUCTOR_API_URL", "https://web-production-c671d.up.railway.app")
CONDUCTOR_API_TOKEN = os.environ.get("CONDUCTOR_API_TOKEN", "")


class CreateLoraFormView(CView):
    """Returns the dialog's form HTML, with the user's selection baked into the submit URL."""

    template_name = "create_lora/form.html"
    permission_classes = [HasAnyRole]
    roles = ["portal_items_read"]

    def get(self, request):
        item_ids = get_item_ids_from_request(request)
        log.info(f"{request.user} opening CreateLora dialog for {len(item_ids)} items")
        ctx = {
            "item_count": len(item_ids),
            # Same selection (query params) carried through to the submit endpoint.
            "submit_link": request.get_full_path().replace(
                "/create_lora/", "/create_lora/submit/"
            ),
        }
        return Response(ctx)


class CreateLoraSubmitView(CView):
    """Reads the selection + form fields and kicks off training via conductor-mcp."""

    permission_classes = [HasAnyRole]
    roles = ["portal_items_read"]

    def post(self, request):
        item_ids = get_item_ids_from_request(request)
        label = request.data.get("label", "").strip()
        trigger_word = request.data.get("trigger_word", "sks").strip() or "sks"

        if not item_ids:
            return JsonResponse({"error": "No items selected."}, status=400)
        if not label:
            return JsonResponse({"error": "Name is required."}, status=400)
        if not CONDUCTOR_API_TOKEN:
            log.error("CONDUCTOR_API_TOKEN is not set on the Portal environment")
            return JsonResponse({"error": "Server is not configured (missing API token)."}, status=500)

        log.info(f"{request.user} starting LoRA training '{label}' on {len(item_ids)} items")

        try:
            resp = requests.post(
                f"{CONDUCTOR_API_URL}/lora/train",
                json={"item_ids": item_ids, "label": label, "trigger_word": trigger_word},
                headers={"Authorization": f"Bearer {CONDUCTOR_API_TOKEN}"},
                timeout=120,  # staging + submitting can take a while; this is a synchronous click
            )
        except requests.RequestException as exc:
            log.exception("Could not reach conductor-mcp")
            return JsonResponse({"error": f"Could not reach training service: {exc}"}, status=502)

        if resp.status_code >= 400:
            return JsonResponse({"error": f"Training service returned {resp.status_code}: {resp.text}"}, status=502)

        return JsonResponse(resp.json())
