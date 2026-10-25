"""Compatibility import for the canonical packaged hardware localizer."""

from __future__ import annotations

import sys

from quantum_simulator import hardware_fault_localizer as _canonical_module

sys.modules[__name__] = _canonical_module
