"""Offline, resumable full-dataset benchmark accounting.

Uses the existing QMutBench parser/diagnoser and the source-backed Bugs4Q
adapter. Unimplemented Bugs4Q source shapes remain explicit unsupported rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from qmutbench_validation import discover_qmutbench_files, validate_qmutbench_case
from bugs4q_integration import OFFICIAL_ROOT, discover_bugs4q_cases, run_bugs4q_integration

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "benchmarks"


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in records if r.get("ground_truth_category") is not None and r.get("predicted_category") is not None]
    tp = sum(r["ground_truth_category"] == "PROGRAM_FAULT" and r["predicted_category"] == "PROGRAM_FAULT" for r in eligible)
    fp = sum(r["ground_truth_category"] != "PROGRAM_FAULT" and r["predicted_category"] == "PROGRAM_FAULT" for r in eligible)
    fn = sum(r["ground_truth_category"] == "PROGRAM_FAULT" and r["predicted_category"] != "PROGRAM_FAULT" for r in eligible)
    tn = sum(r["ground_truth_category"] != "PROGRAM_FAULT" and r["predicted_category"] != "PROGRAM_FAULT" for r in eligible)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    loc = [r for r in records if r.get("ground_truth_location") is not None and r.get("predicted_location") is not None]
    correct = sum(r["ground_truth_location"] == r["predicted_location"] for r in loc)
    return {"evaluated_cases": len(eligible), "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_negatives": tn, "accuracy": (tp + tn) / len(eligible) if eligible else None,
            "precision": precision, "recall": recall, "f1": f1,
            "correct_location": correct, "incorrect_location": len(loc) - correct,
            "localization_accuracy": correct / len(loc) if loc else None,
            "localization_evaluated_cases": len(loc)}


def _summary(name: str, records: list[dict[str, Any]], total: int, discovered: int, parsed: int) -> dict[str, Any]:
    diagnoses = sum(r.get("predicted_category") is not None for r in records)
    unsupported_breakdown: dict[str, int] = {}
    for row in records:
        if row.get("status") != "UNSUPPORTED":
            continue
        reason = str(row.get("unsupported_reason") or next(iter(row.get("errors", [])), "Unspecified unsupported case"))
        unsupported_breakdown[reason] = unsupported_breakdown.get(reason, 0) + 1
    metrics = _metrics(records)
    def breakdown(key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in records:
            value = row.get(key)
            if value is not None:
                label = str(value)
                result[label] = result.get(label, 0) + 1
        return dict(sorted(result.items()))
    return {"benchmark": name, "total_cases": total, "discovered_cases": discovered, "parsed_cases": parsed,
            "executable_cases": sum(r.get("execution_status") == "COMPLETED" for r in records),
            "translated_cases": sum(r.get("translated", r.get("support_status") in {"SUPPORTED", "SUPPORTED_UNCERTAIN", "SUPPORTED_VERIFIABLE"}) for r in records),
            "diagnosed_cases": diagnoses,
            "unsupported_cases": sum(r.get("status") == "UNSUPPORTED" for r in records),
            "error_cases": sum(r.get("status") in {"PARSING_ERROR", "EXECUTION_ERROR", "ERROR"} for r in records),
            "labeled_evaluation_cases": metrics["evaluated_cases"],
            "unsupported_reason_breakdown": unsupported_breakdown,
            "metrics": metrics, "counts_by_fault_category": breakdown("fault_category"),
            "counts_by_mutation_type": breakdown("mutation_type"), "counts_by_gate_type": breakdown("gate_type"),
            "counts_by_num_qubits": breakdown("num_qubits"), "records": records}


def _write(name: str, report: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = name.lower() + "_full_results"
    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rows = report["records"]
    keys = sorted({key for row in rows for key in row})
    with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def run_qmut(resume: bool = True) -> dict[str, Any]:
    dataset = discover_qmutbench_files(ROOT)
    checkpoint = OUT / "qmutbench_full_checkpoint.json"
    saved = {}
    if resume and checkpoint.exists():
        try: saved = {r["case_id"]: r for r in json.loads(checkpoint.read_text(encoding="utf-8"))}
        except (ValueError, OSError): saved = {}
    records = []
    for case in dataset.cases:
        result = saved.get(case.case_id)
        if result is not None and not (result.get("execution_status") == "COMPLETED" and result.get("system_diagnosis")):
            result = None
        if result is None:
            item = validate_qmutbench_case(case).to_dict()
            diagnosis = item.get("system_diagnosis") or {}
            gt = item.get("benchmark_ground_truth") or {}
            result = {"case_id": case.case_id, "source_files": case.source_files,
                      "status": ("PARSING_ERROR" if item.get("support_status") == "PARSING_FAILURE" else "UNSUPPORTED" if item.get("support_status") == "UNSUPPORTED" else "OK"),
                      "support_status": item.get("support_status"),
                      "execution_status": item.get("execution_status"), "predicted_category": diagnosis.get("category"),
                      "ground_truth_category": gt.get("diagnosis_category") if gt.get("diagnosis_category_compatible") else None,
                      "ground_truth_status": gt.get("status"), "mutation_type": (case.raw.get("mutation") or {}).get("operator"),
                      "fault_category": (case.raw.get("mutation") or {}).get("operator"),
                      "gate_type": (case.raw.get("mutation") or {}).get("filename_descriptor"),
                      "num_qubits": case.raw.get("num_qubits"), "predicted_location": None, "ground_truth_location": None,
                      "diagnosis_ground_truth_compatible": item.get("diagnosis_ground_truth_compatible"), "errors": item.get("errors", [])}
            result.update({"parsed": item.get("support_status") != "PARSING_FAILURE",
                           "translated": item.get("supported", False),
                           "executable": item.get("execution_status") == "COMPLETED",
                           "diagnosed": item.get("system_diagnosis") is not None,
                           "system_diagnosis": item.get("system_diagnosis"),
                           "reference_circuit": item.get("reference_circuit"),
                           "mutant_circuit": item.get("mutant_circuit"),
                           "unsupported_reason": (item.get("errors") or [None])[0] if item.get("support_status") == "UNSUPPORTED" else None})
            saved[case.case_id] = result
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(list(saved.values()), indent=2), encoding="utf-8")
        result.setdefault("support_status", "PARSING_FAILURE" if case.parse_error else "UNSUPPORTED" if result.get("status") == "UNSUPPORTED" else "SUPPORTED")
        result.setdefault("status", "PARSING_ERROR" if case.parse_error else "UNSUPPORTED" if result.get("support_status") == "UNSUPPORTED" else "OK")
        records.append(result)
    report = _summary("QMutBench", records, len(dataset.cases), len(dataset.cases), sum(case.parse_error is None for case in dataset.cases))
    report["issues"] = dataset.issues
    report["data_files"] = dataset.data_files
    report["ground_truth_note"] = "Mutation operator is available; no diagnosis-category ground truth or unambiguous fault location is declared by this corpus. Accuracy is therefore undefined."
    _write("qmutbench", report)
    return report


def run_bugs(resume: bool = True) -> dict[str, Any]:
    base = Path(OFFICIAL_ROOT)
    all_dirs = sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    supported = discover_bugs4q_cases(base)
    integration = {r["case_number"]: r for r in run_bugs4q_integration(base).cases}
    records = []
    for directory in all_dirs:
        number = int(directory.name)
        item = integration.get(number)
        if item:
            diag = item.get("system_diagnosis") or {}
            completed = item.get("execution_status") == "COMPLETED"
            item_status = item.get("execution_status")
            records.append({"case_id": f"Bugs4Q-{number}", "status": "OK" if completed else "PARSING_ERROR" if item_status == "PARSING_ERROR" else "UNSUPPORTED", "execution_status": item_status,
                            "predicted_category": diag.get("category"), "ground_truth_category": None,
                            "ground_truth_status": "BUG_TYPE_METADATA_ONLY", "fault_category": item.get("bugs4q_metadata", {}).get("bug_type"),
                            "mutation_type": "source_mutation", "gate_type": None, "num_qubits": len(next(iter(item.get("expected_distribution", {})), "")),
                            "predicted_location": None, "ground_truth_location": None,
                            "errors": item.get("errors", []),
                            "unsupported_reason": item.get("unsupported_reason"),
                            "source_loaded": item.get("source_loaded", True),
                            "circuit_extracted": item.get("circuit_extracted", completed),
                            "translated": item.get("translated", completed),
                            "executable": item.get("executable", completed),
                            "diagnosed": item.get("diagnosed", completed),
                            "reference_fixed_circuit": item.get("reference_fixed_circuit"),
                            "buggy_circuit": item.get("buggy_circuit"),
                            "source_level_mutation": (item.get("bugs4q_metadata") or {}).get("source_level_mutation"),
                            "evaluation_note": "Bug type confirms a bug exists but does not identify a diagnosis category; source-level edit is retained separately."})
        else:
            required = [directory / f"{stem}_{number}.py" for stem in ("buggy", "fixed")]
            missing = [str(p.name) for p in required if not p.is_file()]
            meta = {}
            info = directory / f"info_{number}.csv"
            if info.is_file():
                try:
                    with info.open(encoding="utf-8-sig", newline="") as f: meta = next(csv.DictReader(f, delimiter=";"), {})
                except (OSError, csv.Error): pass
            records.append({"case_id": f"Bugs4Q-{number}", "status": "UNSUPPORTED", "execution_status": "NOT_RUN",
                            "predicted_category": None, "ground_truth_category": None, "ground_truth_status": "BUG_TYPE_METADATA_ONLY",
                            "fault_category": meta.get("Bug-Type"), "mutation_type": "source_mutation", "gate_type": None, "num_qubits": None,
                            "predicted_location": None, "ground_truth_location": None, "errors": ["No general source translator for this case" if not missing else "Missing source files: " + ", ".join(missing)]})
    report = _summary("Bugs4Q", records, len(all_dirs), len(all_dirs), sum(r.get("status") != "PARSING_ERROR" for r in records))
    report["source_pairs_available_cases"] = supported
    report["ground_truth_note"] = "The framework records bug-type metadata and source diffs, not DiagnosisCategory labels. No accuracy is computed from execution success or metadata alone."
    _write("bugs4q", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("qmutbench", "bugs4q", "all"), default="all")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    results = {}
    if args.benchmark in ("qmutbench", "all"): results["qmutbench"] = run_qmut(not args.no_resume)
    if args.benchmark in ("bugs4q", "all"): results["bugs4q"] = run_bugs(not args.no_resume)
    combined_records = [r for report in results.values() for r in report["records"]]
    combined = _summary("Combined", combined_records, sum(r["total_cases"] for r in results.values()),
                        sum(r["discovered_cases"] for r in results.values()), sum(r["parsed_cases"] for r in results.values()))
    combined["benchmarks"] = {key: {k: v for k, v in report.items() if k != "records"} for key, report in results.items()}
    combined["total_discovered"] = combined["discovered_cases"]
    combined["total_executable"] = combined["executable_cases"]
    combined["total_labeled"] = combined["labeled_evaluation_cases"]
    combined["total_unsupported"] = combined["unsupported_cases"]
    combined["total_errors"] = combined["error_cases"]
    combined["combined_metrics"] = combined["metrics"] if combined["labeled_evaluation_cases"] else None
    (OUT / "combined_benchmark_results.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: {key: report[key] for key in ("total_cases", "discovered_cases", "parsed_cases", "executable_cases", "diagnosed_cases", "unsupported_cases", "error_cases", "metrics")} for k, report in results.items()}, indent=2))


if __name__ == "__main__": main()
