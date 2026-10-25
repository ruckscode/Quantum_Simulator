"""Repeated-seed robustness check for the fixed Phase 1 scenario protocol.

This experiment varies only the simulator random seed. Each case retains the
same predeclared circuit, hardware profile, labels, threshold, and 512 shots.
Repeated observations share one deliberately narrow circuit family and are
descriptive sampling checks, not independent generalization evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.fault_isolation_phase1 import (
    build_scenarios,
    case_record,
    summarize,
)
from experiments.run_experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "experiments" / "controlled_fault_isolation_phase1_4"
SCENARIO_PROTOCOL = "controlled_fault_isolation_phase1_v1"
ROBUSTNESS_PROTOCOL = "controlled_fault_isolation_phase1_4_seed_sensitivity_v1"
SHOTS = 512
SEEDS = (20261014, 20261015, 20261016, 20261017, 20261018)


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _scenario_variability(records: list[dict[str, Any]], seed_count: int) -> list[dict[str, Any]]:
    result = []
    scenario_ids = list(dict.fromkeys(record["scenario_id"] for record in records))
    for scenario_id in scenario_ids:
        rows = [record for record in records if record["scenario_id"] == scenario_id]
        tvds = [
            float(row["observable_metrics"]["total_variation_distance"])
            for row in rows
            if row["observable_metrics"]["total_variation_distance"] is not None
        ]
        expected_detection = rows[0]["expected_detection"]
        localizable = [row for row in rows if row["localization_supported"]]
        result.append({
            "scenario_id": scenario_id,
            "ground_truth_class": rows[0]["ground_truth_class"],
            "seed_count": seed_count,
            "detection_rate": (
                _rate([row["anomaly_detected"] is True for row in rows])
                if expected_detection is not None else None
            ),
            "classification_accuracy": _rate([row["category_correct"] for row in rows]),
            "localization_rate": _rate([row["localization_correct"] is True for row in localizable]),
            "localization_trials": len(localizable),
            "abstention_rate": _rate([row["abstained"] for row in rows]),
            "abstentions": sum(row["abstained"] for row in rows),
            "tvd_mean": statistics.fmean(tvds) if tvds else None,
            "tvd_min": min(tvds) if tvds else None,
            "tvd_max": max(tvds) if tvds else None,
        })
    return result


def _seed_summaries(records: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    summaries = []
    for seed in seeds:
        summary = summarize([row for row in records if row["evaluation_seed"] == seed])
        summaries.append({
            "seed": seed,
            "case_count": summary["case_count"],
            "accuracy": summary["classification"]["accuracy"],
            "balanced_accuracy": summary["classification"]["balanced_accuracy"],
            "macro_f1": summary["classification"]["macro_f1"],
            "detection_rate": summary["detection"]["detection_rate"],
            "false_alarm_rate": summary["detection"]["false_alarm_rate"],
            "localization_rate": summary["localization"]["correct_localization_rate"],
            "abstentions": summary["classification"]["abstention_count"],
            "failures": summary["classification"]["failure_count"],
            "unsupported": summary["classification"]["unsupported_count"],
        })
    return summaries


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    records = payload["case_outcomes"]
    json_path = output_dir / "robustness_results.json"
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    case_fields = [
        "scenario_id", "evaluation_seed", "ground_truth_cause", "ground_truth_class",
        "predicted_class", "status", "category_correct", "expected_detection",
        "anomaly_detected", "localization_supported", "localization_correct",
        "abstained", "abstention_reason", "observable_metrics", "settings",
        "program_candidates", "hardware_candidates", "evidence", "missing_evidence",
    ]
    with (output_dir / "scenario_seed_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record[key], sort_keys=True) if isinstance(record[key], (dict, list)) else record[key] for key in case_fields})

    variability_fields = list(payload["scenario_variability"][0])
    with (output_dir / "scenario_variability.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=variability_fields)
        writer.writeheader()
        writer.writerows(payload["scenario_variability"])

    lines = [
        "# Controlled Fault-Isolation — Phase 1.4 Seed Sensitivity",
        "",
        f"Scenario protocol: `{SCENARIO_PROTOCOL}` (the same nine predeclared scenarios and ground-truth rules used in Phase 1).",
        f"Output/evaluation version: `{ROBUSTNESS_PROTOCOL}`. The output version identifies this repeated-seed study; it does not rename the unchanged scenario protocol.",
        "",
        "## Design",
        "",
        f"- Cases: {payload['scenario_count']} scenarios × {len(payload['seeds'])} seeds = {payload['run_count']} runs.",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}. These are five fixed seeds following the Phase 1 seed {payload['baseline_seed']}; the baseline run itself is not included again.",
        f"- Shots: {SHOTS} per case, unchanged from Phase 1. Holding shots fixed isolates sensitivity to seed-dependent sampling; this does not estimate shot-count sensitivity.",
        "- Scenario definitions, hardware profiles, ground-truth labels, and diagnosis threshold are reused unchanged. Ground truth comes from the declared scenario table, not model outputs.",
        "- Repetitions share the same small circuit family. Treat rates as descriptive repeated-sampling results, not independent observations or evidence of generalization.",
        "- Execution uses the local simulator only; this is not physical-hardware validation.",
        "",
        "## Aggregate results across scenario-seed runs",
        "",
        f"- Statuses: {payload['status_counts']}",
        f"- Accuracy: {payload['classification']['accuracy']:.3f}; balanced accuracy: {payload['classification']['balanced_accuracy']:.3f}; macro-F1: {payload['classification']['macro_f1']:.3f}.",
        f"- Detection: {payload['detection']['detected_positive_cases']}/{payload['detection']['expected_positive_cases']} ({payload['detection']['detection_rate']:.3f}); false alarms {payload['detection']['false_alarms']}/{payload['detection']['expected_negative_cases']}.",
        f"- Localization: {payload['localization']['cases_with_all_targets_found']}/{payload['localization']['cases_with_expected_targets']} supported target-cases ({payload['localization']['correct_localization_rate']:.3f}).",
        f"- Abstentions: {payload['classification']['abstention_count']}; failures: {payload['classification']['failure_count']}; unsupported: {payload['classification']['unsupported_count']}.",
        "",
        "## Per-seed metrics",
        "",
        "| Seed | Accuracy | Balanced accuracy | Macro-F1 | Detection | False alarms | Localization | Abstentions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_summaries"]:
        lines.append(
            f"| {row['seed']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['detection_rate']:.3f} | {row['false_alarm_rate']:.3f} | {row['localization_rate']:.3f} | {row['abstentions']} |"
        )
    lines.extend([
        "",
        "## Per-scenario variability",
        "",
        "| Scenario | Truth | Detection rate | Classification accuracy | Localization rate (n) | Abstentions | TVD mean [min, max] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["scenario_variability"]:
        detection = "n/a" if row["detection_rate"] is None else f"{row['detection_rate']:.3f}"
        localization = "n/a" if row["localization_rate"] is None else f"{row['localization_rate']:.3f} ({row['localization_trials']})"
        tvd = "n/a" if row["tvd_mean"] is None else f"{row['tvd_mean']:.4f} [{row['tvd_min']:.4f}, {row['tvd_max']:.4f}]"
        lines.append(
            f"| {row['scenario_id']} | {row['ground_truth_class']} | {detection} | "
            f"{row['classification_accuracy']:.3f} | {localization} | "
            f"{row['abstentions']}/{row['seed_count']} | {tvd} |"
        )
    lines.extend([
        "",
        "Full per-scenario/per-seed outcomes, evidence, settings, and candidates are in `scenario_seed_results.csv` and the case JSON files. `scenario_variability.csv` and `robustness_results.json` preserve the same summaries and provenance.",
        "",
        "## Limits",
        "",
        "Five fixed seeds at one shot count quantify only seed sensitivity for these constructed scenarios. The nine scenarios share one circuit family; no confidence interval or generalization claim is warranted. Ground truth is scenario-defined, and all results are simulator-derived.",
        "",
    ])
    with (output_dir / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_robustness_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    scenarios = build_scenarios()
    if any(scenario.config.shots != SHOTS for scenario in scenarios):
        raise ValueError("Phase 1 scenario shot counts changed; refusing a non-comparable robustness run")

    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for scenario in scenarios:
            config = replace(
                scenario.config,
                random_seed=seed,
                experiment_name=f"{scenario.config.experiment_name}_seed_{seed}",
            )
            seeded_scenario = replace(scenario, config=config)
            case_path = output_dir / "cases" / f"seed_{seed}" / f"{scenario.scenario_id}.json"
            case_path.parent.mkdir(parents=True, exist_ok=True)
            result = run_experiment(config, output_path=case_path)
            record = case_record(seeded_scenario, result)
            record["evaluation_seed"] = seed
            if record["ground_truth_class"] != scenario.ground_truth_class:
                raise AssertionError(f"Ground-truth drift for {scenario.scenario_id} at seed {seed}")
            records.append(record)

    aggregate = summarize(records)
    variability = _scenario_variability(records, len(SEEDS))
    payload = {
        **aggregate,
        "protocol": {
            "name": ROBUSTNESS_PROTOCOL,
            "scenario_protocol_name": SCENARIO_PROTOCOL,
            "scenario_protocol_source": "experiments/fault_isolation_phase1.py::build_scenarios",
            "reference_phase1_output_dir": "reports/experiments/controlled_fault_isolation_phase1",
            "output_version": "phase1_4",
            "execution": "local simulator via experiments.run_experiment.run_experiment",
            "shots_per_case": SHOTS,
            "seeds": list(SEEDS),
            "baseline_seed_not_repeated": 20261013,
            "seed_schedule_rationale": "Five new fixed seeds quantify seed-dependent sampling variation while preserving 512 shots per scenario.",
            "ground_truth_source": "predeclared scenario cause/component in experiments.fault_isolation_phase1.build_scenarios; never inferred from diagnosis output",
            "unit_of_analysis": "scenario-seed run; repeated runs share scenario and circuit family and are not independent generalization samples",
            "physical_hardware_validation": False,
            "scenario_count": len(scenarios),
            "run_count": len(records),
        },
        "scenario_count": len(scenarios),
        "run_count": len(records),
        "seeds": list(SEEDS),
        "shots_per_case": SHOTS,
        "baseline_seed": 20261013,
        "seed_summaries": _seed_summaries(records, SEEDS),
        "scenario_variability": variability,
    }
    _write_outputs(output_dir, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run_robustness_experiment(args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "scenario_count": report["scenario_count"],
        "run_count": report["run_count"],
        "accuracy": report["classification"]["accuracy"],
        "macro_f1": report["classification"]["macro_f1"],
        "detection_rate": report["detection"]["detection_rate"],
        "localization_rate": report["localization"]["correct_localization_rate"],
        "abstentions": report["classification"]["abstention_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
