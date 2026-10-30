"""Run a bounded, larger development-only classifier-family comparison."""
from pathlib import Path

from qmutbench_qml.training.pilot import run


if __name__ == "__main__":
    root = Path.cwd()
    output = root / "qmutbench_qml/results/dev_expansion_24origins_20261026"
    result = run(root, output, origin_count=24, rows_per_origin=12,
                 include_data_reuploading=True, include_quantum_reservoir=True)
    print(f"Expansion complete: {result['sampled_records']} rows across {len(result['selected_origins'])} development origins")
    print(f"Results: {output / 'research_report.md'}")
