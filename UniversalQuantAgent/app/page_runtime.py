"""Stable access to shared Streamlit render helpers.

Streamlit executes navigation pages as ``__main__`` modules. Importing the
entry script from those pages can therefore resolve to a transient page module
instead of the dashboard host. This adapter first uses the host module saved
by ``app.app.main`` and only imports ``app.app`` when a page is tested alone.
"""

from __future__ import annotations

import sys
from types import ModuleType


def _runtime() -> ModuleType:
    host = sys.modules.get("_uqa_app_runtime")
    if host is not None:
        return host

    from app import app as standalone_host

    return standalone_host


def __getattr__(name: str):
    """Forward requested presentation helpers to the stable app host."""
    return getattr(_runtime(), name)