"""
Registers the plugins for the "Create LoRA" action.

Adapted from Cantemo's own reference implementation, SearchPageExportExample
(https://github.com/Cantemo/SearchPageExportExample) -- same shape, "an
action based on user selection on the search page," just pointed at our
own form/view instead of a CSV export.
"""
import logging

from portal.generic.plugin_interfaces import IAppRegister
from portal.generic.plugin_interfaces import IPluginBlock
from portal.generic.plugin_interfaces import IPluginURL
from portal.pluginbase.core import Plugin
from portal.pluginbase.core import implements

log = logging.getLogger(__name__)


class CreateLoraPluginURL(Plugin):
    """Registers the URL endpoints defined in urls.py under /create_lora/."""

    implements(IPluginURL)

    def __init__(self):
        self.name = "CreateLora App"
        self.urls = "portal.plugins.create_lora.urls"
        self.urlpattern = r"^create_lora/"
        self.namespace = "create_lora"
        # Generated once for this plugin -- do not reuse elsewhere.
        self.plugin_guid = "6f2a8e1c-6a2f-4b8f-9a2d-1f6d2c4e8b17"
        log.debug("Initiated CreateLoraPluginURL")


CreateLoraPluginURL()


class CreateLoraRegister(Plugin):
    """Adds the app to Portal's registered-apps admin listing."""

    implements(IAppRegister)

    def __init__(self):
        self.name = "CreateLora Registration App"
        self.plugin_guid = "9b3c7d2e-4f1a-4e6b-8c5d-2a7e9f0b3c46"
        log.debug("Registered the CreateLora App")

    def __call__(self):
        from .__init__ import __version__ as versionnumber

        return {
            "name": "CreateLora",
            "version": versionnumber,
            "author": "Entertainment Technologists Inc.",
            "author_url": "https://www.entertainmentconsultancy.com/",
            "notes": "CoreWeave IBC 2026 demo -- train a LoRA from selected MAM assets.",
        }


CreateLoraRegister()


class CreateLoraSearchMenuPlugin(Plugin):
    """
    Adds "Create LoRA" to the search page's multi-select action menu.

    Block name is Portal's own name for this menu -- see
    https://doc.cantemo.com/latest/DevelopmentGuide/modules/search_results.html
    "Large Gear Box ... rendered after the actions on selected items."
    """

    implements(IPluginBlock)

    def __init__(self):
        self.name = "vs_collection_view_dropdown"
        self.plugin_guid = "2d8f4a6b-1c3e-4f7a-9b2d-5e8c1a4f7d29"

    def return_string(self, tagname, *args):
        return {"guid": self.plugin_guid, "template": "create_lora/menu_item.html"}


CreateLoraSearchMenuPlugin()


class CreateLoraSearchPageJavascriptPlugin(Plugin):
    """Injects the JavaScript the menu item's onclick handler needs."""

    implements(IPluginBlock)

    def __init__(self):
        self.name = "new_search_view_script"
        self.plugin_guid = "7a1e4c9d-3b6f-4a8e-9d1c-4f7b2e8a6c53"

    def return_string(self, tagname, *args):
        return {"guid": self.plugin_guid, "template": "create_lora/search_page_javascript.html"}


CreateLoraSearchPageJavascriptPlugin()
