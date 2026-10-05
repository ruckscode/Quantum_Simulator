"""Compatibility entry point that exposes the packaged simulator locally.

When Python starts from the repository root, this file takes precedence over
the package in ``src``. Load that package under its canonical name so local
imports and installed imports expose the same implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_package_dir = Path(__file__).resolve().parent / "src" / "quantum_simulator"
_package_init = _package_dir / "__init__.py"
_package_spec = importlib.util.spec_from_file_location(
    __name__,
    _package_init,
    submodule_search_locations=[str(_package_dir)],
)
if _package_spec is None or _package_spec.loader is None:
    raise ImportError(f"Could not load the packaged module from {_package_init}")

_package = importlib.util.module_from_spec(_package_spec)
sys.modules[__name__] = _package
_package_spec.loader.exec_module(_package)
