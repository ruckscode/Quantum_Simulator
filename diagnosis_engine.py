"""Compatibility import for the canonical packaged diagnosis engine."""

from __future__ import annotations

import sys

from quantum_simulator import diagnosis_engine as _canonical_module

sys.modules[__name__] = _canonical_module
