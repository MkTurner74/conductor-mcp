__version__ = "0.1.0"

# Registers the plugins in plugin.py with Cantemo's PluginEnvironment on
# app load -- required, mirrors every other Portal app.
from . import plugin  # noqa: F401
