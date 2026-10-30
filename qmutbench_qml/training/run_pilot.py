"""CLI entry point for the bounded QMutBench QML pilot."""
from pathlib import Path

from qmutbench_qml.training.pilot import run


if __name__ == "__main__":
    project_root = Path.cwd()
    result = run(project_root)
    print(f"Pilot complete: {result['sampled_records']} rows across {len(result['selected_origins'])} origins")
    print(f"Results: {project_root / 'qmutbench_qml/results/research_report.md'}")
