"""Common orchestration and report structures for independent benchmarks.

Benchmark adapters retain their own case semantics and metrics. This module
coordinates reports without combining labels or modifying diagnosis behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class NormalizedBenchmarkCase:
    benchmark_name: str
    case_id: str
    source_files: list[str] = field(default_factory=list)
    reference_circuit: Any = None
    mutant_circuit: Any = None
    expected_information: Any = None
    source_references: list[Any] = field(default_factory=list)
    ground_truth: Any = None
    ground_truth_status: str = "MISSING"
    mutation_metadata: dict[str, Any] | None = None
    source_level_information: dict[str, Any] = field(default_factory=dict)
    benchmark_metadata: dict[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    benchmark_name: str

    def run(self, root: str | Path) -> Mapping[str, Any]: ...


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class Bugs4QBenchmarkAdapter:
    """Thin wrapper around the existing Bugs4Q loader and validator."""

    benchmark_name = "Bugs4Q"

    def run(self, root: str | Path) -> Mapping[str, Any]:
        from bugs4q_validation import load_bugs4q_dataset, validate_bugs4q_dataset

        dataset = load_bugs4q_dataset(root)
        return validate_bugs4q_dataset(dataset).to_dict()


def _benchmark_summary(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    if name == "Bugs4Q":
        cases = int(report.get("cases_discovered", 0))
        supported = int(report.get("supported", 0))
        unsupported = int(report.get("unsupported", 0))
        uncertain = int(report.get("uncertain_or_conflicting_ground_truth_cases", 0))
        reliable = int(report.get("reliable_ground_truth_cases", 0))
        parsing_failures = int(report.get("dataset_file_parse_failures", 0)) + sum(
            record.get("support_status") == "PARSING_FAILURE" for record in report.get("case_level_results", [])
        )
        status = "PARTIAL / DATASET_UNAVAILABLE" if not report.get("dataset_found") else "EVALUATED"
    else:
        cases = int(report.get("total_cases", 0))
        supported = int(report.get("supported_cases", 0))
        unsupported = int(report.get("unsupported_cases", 0))
        uncertain = int(report.get("uncertain_cases", 0)) + int(report.get("conflicting_cases", 0)) + int(report.get("missing_ground_truth_cases", 0))
        reliable = int(report.get("reliable_ground_truth_cases", 0))
        parsing_failures = int(report.get("parsing_failures", 0))
        status = str(report.get("status", "NOT_EVALUATED"))

    summary = {
        "status": status,
        "cases": cases,
        "supported": supported,
        "unsupported": unsupported,
        "uncertain": uncertain,
        "reliable_ground_truth": reliable,
        "parsing_failures": parsing_failures,
        "metrics": report.get("metrics", {}),
    }
    if name != "Bugs4Q":
        summary["dataset_type"] = report.get("dataset_type", "UNKNOWN")
    return summary


def run_benchmark_validation(
    root: str | Path,
    *,
    report_directory: str | Path | None = None,
    adapters: Sequence[BenchmarkAdapter] | None = None,
) -> dict[str, Any]:
    """Run each adapter independently and create individual plus combined reports."""

    root_path = Path(root)
    destination = Path(report_directory) if report_directory is not None else root_path / "reports" / "benchmarks"
    if adapters is None:
        from qmutbench_validation import QMutBenchAdapter

        adapters = (Bugs4QBenchmarkAdapter(), QMutBenchAdapter())

    reports: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    for adapter in adapters:
        name = adapter.benchmark_name
        try:
            reports[name] = dict(adapter.run(root_path))
        except Exception as exc:
            reports[name] = {
                "benchmark": name,
                "status": "ADAPTER_FAILURE",
                "total_cases": 0,
                "supported_cases": 0,
                "unsupported_cases": 0,
                "uncertain_cases": 0,
                "conflicting_cases": 0,
                "missing_ground_truth_cases": 0,
                "reliable_ground_truth_cases": 0,
                "parsing_failures": 0,
                "cases": [],
                "metrics": {"accuracy": None, "reason": "Adapter failed before producing case results."},
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            issues.append({"benchmark": name, "error": f"{type(exc).__name__}: {exc}"})
        slug = name.lower().replace("&", "and").replace(" ", "_")
        _write_json(destination / f"{slug}_validation_report.json", reports[name])

    summaries = {name: _benchmark_summary(name, report) for name, report in reports.items()}
    combined = {
        "report_name": "Independent Multi-Benchmark Validation Summary",
        "dataset_root": str(root_path),
        "benchmark_summary": summaries,
        "benchmark_reports": {
            name: f"{name.lower().replace('&', 'and').replace(' ', '_')}_validation_report.json"
            for name in reports
        },
        "combined_metrics": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "reason": "Benchmark ground-truth definitions are not presumed compatible; metrics remain benchmark-specific.",
        },
        "errors": issues,
        "notes": [
            "Bugs4Q and QMutBench are evaluated and reported independently.",
            "Synthetic test fixtures are not benchmark cases or ground truth.",
            "Mutation metadata is external benchmark context and is not supplied to the diagnosis engine as a label.",
        ],
    }
    _write_json(destination / "combined_validation_report.json", combined)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Run available benchmark adapters independently.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--report-directory")
    args = parser.parse_args()
    combined = run_benchmark_validation(args.root, report_directory=args.report_directory)
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0 if not combined["errors"] else 1


__all__ = [
    "BenchmarkAdapter",
    "Bugs4QBenchmarkAdapter",
    "NormalizedBenchmarkCase",
    "run_benchmark_validation",
]


if __name__ == "__main__":
    raise SystemExit(main())
