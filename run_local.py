#!/usr/bin/env python3
"""Safe local launcher for the Universal Quant Agent Streamlit dashboard.

Running ``streamlit run`` from inside ``UniversalQuantAgent/app`` makes Python
treat ``app/app.py`` as a top-level module named ``app``. That shadows the real
``app`` package, so ``from app.style import ...`` raises::

    ModuleNotFoundError: No module named 'app.style'; 'app' is not a package

This launcher removes the ambiguity. It always starts Streamlit from the repo
root, forces the repo root and the ``UniversalQuantAgent`` project root to the
front of ``sys.path``, and purges any already-imported non-package ``app``
shadow before handing off to Streamlit -- so the import path is correct no
matter which directory you invoke it from.

Usage::

    python run_local.py                 # launch the dashboard
    python run_local.py --server.port 8600   # extra args pass through to Streamlit

See docs/local_launch.md for the full explanation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT / "UniversalQuantAgent"
APP_ENTRY = PROJECT_ROOT / "app" / "app.py"


def _harden_import_path() -> None:
    """Put the correct roots first on ``sys.path`` and drop a shadowed ``app``."""
    for entry in (str(REPO_ROOT), str(PROJECT_ROOT)):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    shadowed = sys.modules.get("app")
    if shadowed is not None and not hasattr(shadowed, "__path__"):
        del sys.modules["app"]


def main() -> int:
    if not APP_ENTRY.is_file():
        print(f"Cannot find the app entry script: {APP_ENTRY}", file=sys.stderr)
        return 1

    _harden_import_path()
    # Streamlit reads .streamlit/config.toml from the launch directory; the repo
    # root is where that file lives, matching the Render start command.
    os.chdir(REPO_ROOT)

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(APP_ENTRY), *sys.argv[1:]]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
