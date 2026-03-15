"""Resolve paths to bundled Rust binaries.

Works in both development mode (project root) and when packaged
with Briefcase (macOS .app bundle).  In both cases the binaries
live at ``<ancestors[2]>/resources/bin/`` relative to this file:

- Dev:       src/chunk_embed/_paths.py  →  parents[2] = project root
- Briefcase: Resources/app/chunk_embed/_paths.py  →  parents[2] = Resources/
"""

from __future__ import annotations

import os
from pathlib import Path

_BUNDLED_BIN = Path(__file__).resolve().parents[2] / "resources" / "bin"


def bundled_bin_dir() -> Path | None:
    """Return the directory containing bundled Rust binaries, or None."""
    if _BUNDLED_BIN.is_dir():
        return _BUNDLED_BIN
    return None


def prepend_bundled_bin_to_path() -> None:
    """Add the bundled binary directory to PATH (if it exists)."""
    bin_dir = bundled_bin_dir()
    if bin_dir is not None:
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
