"""Compatibility import for the canonical packaged gate registry."""

from __future__ import annotations

import sys

from quantum_simulator import gates as _canonical_module

sys.modules[__name__] = _canonical_module
