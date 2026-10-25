"""Compatibility module combining stdlib statistics and simulator helpers.

This repository-root module shadows Python's standard-library module when the
project root is on ``sys.path``. Load the standard-library implementation into
this module's namespace, then add the simulator's comparison helpers so both
existing import styles continue to work.
"""

from pathlib import Path
import sysconfig


_stdlib_statistics = Path(sysconfig.get_path("stdlib")) / "statistics.py"
exec(compile(_stdlib_statistics.read_bytes(), str(_stdlib_statistics), "exec"), globals())

from quantum_simulator.statistics import (  # noqa: E402
    DistributionComparisonResult,
    compare_counts,
    compare_distributions,
)

__all__ = list(dict.fromkeys([
    *globals().get("__all__", ()),
    "DistributionComparisonResult",
    "compare_counts",
    "compare_distributions",
]))
