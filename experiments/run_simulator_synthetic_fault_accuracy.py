"""Small seeded synthetic smoke benchmark for the classical diagnosis path.

The cases are controlled by construction and are not independent hardware
measurements or deployment accuracy evidence. This script avoids QML entirely.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_circuit  # noqa: E402
from quantum_simulator.hardware_model import HardwareModel  # noqa: E402

OUTPUT = ROOT / "reports" / "experiments" / "classical_simulator_fault_accuracy_20261030.json"
SHOTS = 1024
SEEDS = tuple(range(10))

TEMPLATES = (
    (
        [("x", 0), ("measure", [0])],
        [("h", 0), ("measure", [0])],
        1,
    ),
    (
        [("h", 0), ("cx", 0, 1), ("measure", [0, 1])],
        [("x", 0), ("cx", 0, 1), ("measure", [0, 1])],
        2,
    ),
)


def hardware_model(*, degraded: bool) -> HardwareModel:
    model = HardwareModel(2, backend_name="synthetic_backend_profile")
    model.calibration_timestamp_provenance = "synthetic"
    for qubit in range(2):
        model.set_qubit_properties(
            qubit,
            t1=240.0,
            t2=170.0,
            readout_error=0.08 if degraded and qubit == 0 else 0.012,
        )
        for gate in ("h", "x", "cx"):
            qubits = (qubit,) if gate != "cx" else (0, 1)
            error = 0.06 if degraded and gate == "cx" else 0.008
            model.add_gate_calibration(gate, qubits=qubits, error=error, duration=0.05)
    return model


def run_case(label: str, template_index: int, seed: int) -> dict:
    reference, faulty, qubits = TEMPLATES[template_index]
    actual = faulty if label in {"PROGRAM_FAULT", "BOTH"} else reference
    degraded = label in {"HARDWARE_ANOMALY", "BOTH"}
    result = diagnose_circuit(
        actual,
        reference_circuit=reference,
        hardware_model=hardware_model(degraded=degraded),
        shots=SHOTS,
        seed=seed + 100 * template_index,
    )
    expected = DiagnosisCategory.AMBIGUOUS.value if label == "BOTH" else label
    predicted = result.category.value
    correct = predicted == expected
    return {
        "scenario": label,
        "template": f"{qubits}_qubit",
        "seed": seed,
        "expected_category": expected,
        "predicted_category": predicted,
        "correct": correct,
        "anomaly_detected": result.anomaly_detected,
        "tvd": None if result.statistics_result is None else result.statistics_result.total_variation_distance,
        "p_value": getattr(result.statistics_result, "p_value", None),
        "program_evidence_count": len(result.program_evidence_candidates),
        "hardware_evidence_count": len(result.hardware_evidence_candidates),
        "summary": result.summary,
    }


def main() -> None:
    records = [
        run_case(label, template_index, seed)
        for template_index in range(len(TEMPLATES))
        for seed in SEEDS
        for label in ("NO_ANOMALY", "PROGRAM_FAULT", "HARDWARE_ANOMALY", "BOTH")
    ]
    by_class: dict[str, list[dict]] = defaultdict(list)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        by_class[row["scenario"]].append(row)
        confusion[row["expected_category"]][row["predicted_category"]] += 1
    summary = {
        "case_count": len(records),
        "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "per_scenario": {
            key: {
                "correct": sum(row["correct"] for row in values),
                "total": len(values),
                "accuracy": sum(row["correct"] for row in values) / len(values),
            }
            for key, values in sorted(by_class.items())
        },
        "confusion": {key: dict(value) for key, value in sorted(confusion.items())},
    }
    report = {
        "benchmark": "classical_simulator_fault_accuracy_controlled_synthetic_v1",
        "scope": "classical simulator diagnosis only; QML excluded",
        "synthetic_only": True,
        "independent_hardware_data": False,
        "deployment_accuracy_claim": False,
        "shots_per_case": SHOTS,
        "seeds": list(SEEDS),
        "hardware_profile": {
            "baseline_t1_us": 240.0,
            "baseline_t2_us": 170.0,
            "baseline_readout_error": 0.012,
            "degraded_readout_error": 0.08,
            "degraded_cx_error": 0.06,
        },
        "summary": summary,
        "records": records,
        "limitations": [
            "Synthetic cases reuse two hand-authored circuit templates and known injected causes.",
            "Ten seeds per template and cause are repeated simulation trials, not independent device runs.",
            "Calibration values are plausible controlled inputs, not measured calibration snapshots.",
            "The ML classifier is reported separately as an unvalidated advisory and is not scored as the adjudicator.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
