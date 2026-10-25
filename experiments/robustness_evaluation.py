"""Seeded robustness sweep for the deterministic diagnosis/localization pipeline.

This evaluation intentionally does not import or invoke an ML classifier.
Circuits are generated from GHZ, parity, and QFT-like families, which are not
the fixed examples used by the existing Stage 3 tests.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_execution
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "robustness_evaluation_report.md"
SHOTS = 128
_IDEAL_CACHE: dict[tuple[Any, ...], dict[str, float]] = {}
SEEDS = (11, 29, 47)
WIDTHS = (("small", 2), ("medium", 3), ("larger", 4))
DEPTHS = (2, 4)
STRENGTHS = (0.15, 0.45, 0.8)


def _family_circuit(family: str, n: int, depth: int, seed: int) -> list[tuple[Any, ...]]:
    """Produce an unseen circuit family; measurements cover every qubit."""
    ops: list[tuple[Any, ...]] = []
    phase = (seed % 7 + 1) * 0.071
    if family == "ghz":
        ops.append(("h", 0))
        for _ in range(depth):
            for q in range(n - 1):
                ops.append(("cx", q, q + 1))
            ops.append(("rz", (depth % n), phase))
    elif family == "parity":
        for layer in range(depth):
            for q in range(n):
                ops.append(("ry", q, phase * (layer + 1)))
            for q in range(0, n - 1, 2):
                ops.append(("cz", q, q + 1))
            for q in range(1, n - 1, 2):
                ops.append(("cx", q, q + 1))
    elif family == "qft_like":
        for layer in range(depth):
            for q in range(n):
                ops.append(("h", q))
                if q + 1 < n:
                    ops.append(("cp", q, q + 1, phase / (layer + 1)))
            ops.append(("rx", layer % n, phase))
    else:
        raise ValueError(f"unknown circuit family: {family}")
    ops.append(("measure", tuple(range(n))))
    return ops


def _program_fault(circuit: list[tuple[Any, ...]], kind: str) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    body = list(circuit[:-1])
    index = max(0, len(body) // 2)
    original = body[index]
    if kind == "gate_replacement":
        replacement = {"h": "x", "x": "h", "cx": "cz", "cz": "cx", "ry": "rx", "rx": "ry", "cp": "crz", "rz": "rx"}.get(original[0], "x")
        changed = (replacement, *original[1:])
        body[index] = changed
        target = changed
    elif kind == "gate_deletion":
        target = original
        del body[index]
    elif kind == "gate_insertion":
        changed = ("x", 0)
        body.insert(index, changed)
        target = changed
    elif kind == "parameter_modification":
        parameterized = next((i for i, op in enumerate(body) if op[0] in {"rx", "ry", "rz", "cp", "crz"}), None)
        if parameterized is None:
            raise ValueError("generated circuit lacks parameterized operations")
        index = parameterized
        original = body[index]
        changed = (*original[:-1], float(original[-1]) + math.pi * 0.7)
        body[index] = changed
        target = changed
    elif kind == "single_qubit_fault":
        index = next(i for i, op in enumerate(body) if len(_operation_qubits(op)) == 1)
        original = body[index]
        changed = ({"h": "x", "x": "h", "ry": "rx", "rx": "rz", "rz": "ry"}.get(original[0], "x"), *original[1:])
        body[index] = changed
        target = changed
    elif kind == "two_qubit_fault":
        index = next(i for i, op in enumerate(body) if len(_operation_qubits(op)) == 2)
        original = body[index]
        changed = ({"cx": "cz", "cz": "cx", "cp": "crz", "crz": "cp"}.get(original[0], "cz"), *original[1:])
        body[index] = changed
        target = changed
    else:
        raise ValueError(f"unknown program fault: {kind}")
    return body + [circuit[-1]], {"operation": target[0], "qubits": _operation_qubits(target), "index": index}


def _operation_qubits(operation: tuple[Any, ...]) -> tuple[int, ...]:
    name = operation[0]
    count = 2 if name in {"cx", "cz", "cp", "crz", "swap"} else 1
    return tuple(int(q) for q in operation[1:1 + count] if isinstance(q, int))


def _hardware(kind: str, n: int, strength: float, circuit: list[tuple[Any, ...]]) -> tuple[HardwareModel, dict[str, Any]]:
    model = HardwareModel(n, backend_name=f"robust_{kind}_{strength}")
    affected = 0
    if kind in {"readout_error", "combined"}:
        model.set_qubit_properties(affected, readout_error=strength)
    if kind in {"gate_error", "combined"}:
        for op in circuit:
            qubits = _operation_qubits(op)
            if affected in qubits and op[0] != "measure":
                model.add_gate_calibration(op[0], qubits=qubits, error=strength, duration=0.0)
    if kind in {"t1_degradation", "combined"}:
        t1 = max(0.0001, 100.0 * (1.0 - strength))
        model.set_qubit_properties(affected, t1=t1)
        for op in circuit:
            qubits = _operation_qubits(op)
            if affected in qubits and op[0] != "measure":
                model.add_gate_calibration(op[0], qubits=qubits, error=(strength if kind == "combined" else 0.0), duration=100.0)
    if kind == "t2_degradation":
        model.set_qubit_properties(affected, t2=max(0.0001, 100.0 * (1.0 - strength)))
        for op in circuit:
            qubits = _operation_qubits(op)
            if affected in qubits and op[0] != "measure":
                model.add_gate_calibration(op[0], qubits=qubits, duration=100.0)
    return model, {"component_type": "qubit", "component": affected, "strength": strength, "kind": kind}


def _distribution(circuit: list[tuple[Any, ...]], n: int, seed: int, hardware: HardwareModel | None) -> dict[str, float]:
    counts = QuantumSimulator(n, seed=seed).run_shots(circuit, shots=SHOTS, seed=seed, hardware_model=hardware)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    reference = case["reference"]
    observed_program = case["observed_program"]
    n, seed = case["n_qubits"], case["seed"]
    model = case["hardware"]
    cache_key = (n, seed, tuple(tuple(op) for op in reference))
    if cache_key not in _IDEAL_CACHE:
        _IDEAL_CACHE[cache_key] = _distribution(reference, n, seed, None)
    expected = dict(_IDEAL_CACHE[cache_key])
    execution_model = model if case["scenario"] == "HARDWARE_ANOMALY" else None
    observed = _distribution(observed_program, n, seed, execution_model)
    keys = set(expected) | set(observed)
    expected = {key: expected.get(key, 0.0) for key in keys}
    observed = {key: observed.get(key, 0.0) for key in keys}
    _, _, _, trace = QuantumSimulator(n, seed=seed).run_circuit_traced(reference, observed_program, hardware_model=execution_model)
    result = diagnose_execution(expected, observed, expected_circuit=reference, observed_circuit=observed_program,
        hardware_model=model, shots=SHOTS, precomputed_trace=trace)
    if case["scenario"] == "NORMAL":
        anomaly_truth = False
    else:
        anomaly_truth = True
    return {
        "scenario": case["scenario"], "fault_type": case["fault_type"], "hardware_type": case["hardware_type"],
        "family": case["family"], "size": case["size"], "n_qubits": n, "depth": case["depth"],
        "seed": seed, "strength": case["strength"], "truth": case["truth"], "anomaly_truth": anomaly_truth,
        "anomaly_predicted": result.anomaly_detected,
        "category_truth": "NO_ANOMALY" if case["scenario"] == "NORMAL" else case["scenario"],
        "category_predicted": result.category.value,
        "program_candidates": [{"operation": c.operation, "qubits": list(c.qubits), "index": c.index, "reason": c.reason}
                               for c in (result.program_result.candidates if result.program_result else [])],
        "program_evidence": [{"operation": c.operation, "qubits": list(c.qubits), "index": c.index}
                             for c in result.program_evidence_candidates],
        "hardware_candidates": [{"component_type": c.component_type, "component": list(c.component) if isinstance(c.component, tuple) else c.component,
                                 "score": c.score, "reason": c.reason}
                                for c in result.hardware_evidence_candidates],
        "hardware_result_candidates": [{"component_type": c.component_type, "component": list(c.component) if isinstance(c.component, tuple) else c.component,
                                        "score": c.score, "reason": c.reason}
                                       for c in (result.hardware_result.candidates if result.hardware_result else [])],
        "missing_evidence": result.missing_evidence,
    }


def generate_cases() -> list[dict[str, Any]]:
    cases = []
    families = ("ghz", "parity", "qft_like")
    program_types = ("gate_replacement", "gate_deletion", "gate_insertion", "parameter_modification", "single_qubit_fault", "two_qubit_fault")
    hardware_types = ("readout_error", "gate_error", "t1_degradation", "t2_degradation", "combined")
    for size, n in WIDTHS:
        for depth in DEPTHS:
            for family in families:
                for seed in SEEDS:
                    ref = _family_circuit(family, n, depth, seed)
                    common = {"family": family, "size": size, "n_qubits": n, "depth": depth, "seed": seed,
                              "reference": ref, "observed_program": ref, "hardware": HardwareModel(n), "strength": 0.0}
                    cases.append({**common, "scenario": "NORMAL", "fault_type": "none", "hardware_type": "healthy", "truth": None})
                    for kind in program_types:
                        # Program fault types alternate across single and two-qubit operations naturally
                        # as their target operation is selected from the family circuit.
                        observed, truth = _program_fault(ref, kind)
                        cases.append({**common, "scenario": "PROGRAM_FAULT", "fault_type": kind,
                                      "hardware_type": "healthy", "observed_program": observed, "truth": truth})
                    for kind in hardware_types:
                        for strength in STRENGTHS:
                            model, truth = _hardware(kind, n, strength, ref)
                            if kind == "gate_error":
                                target_gate = next(op for op in ref if 0 in _operation_qubits(op) and op[0] != "measure")
                                truth = {"component_type": "gate", "component": list(_operation_qubits(target_gate)), "gate": target_gate[0], "strength": strength, "kind": kind}
                            cases.append({**common, "scenario": "HARDWARE_ANOMALY", "fault_type": "none",
                                          "hardware_type": kind, "hardware": model, "strength": strength, "truth": truth})
    return cases


def evaluate() -> list[dict[str, Any]]:
    return [_run_case(case) for case in generate_cases()]


def _rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    return {
        "cases": total,
        "anomaly_correct": sum(r["anomaly_truth"] == r["anomaly_predicted"] for r in records),
        "routing_correct": sum(r["category_truth"] == r["category_predicted"] for r in records),
        "program_localized": sum(any(c["operation"] == r["truth"]["operation"] and c["qubits"] == list(r["truth"]["qubits"]) for c in r["program_evidence"]) for r in records if r["scenario"] == "PROGRAM_FAULT"),
        "program_cases": sum(r["scenario"] == "PROGRAM_FAULT" for r in records),
        "hardware_localized": sum(any(c["component_type"] == r["truth"]["component_type"] and c["component"] == r["truth"]["component"] for c in r["hardware_candidates"]) for r in records if r["scenario"] == "HARDWARE_ANOMALY"),
        "hardware_cases": sum(r["scenario"] == "HARDWARE_ANOMALY" for r in records),
        "end_to_end_correct": sum(r["category_truth"] == r["category_predicted"] and
                                   (r["scenario"] != "PROGRAM_FAULT" or any(c["operation"] == r["truth"]["operation"] and c["qubits"] == list(r["truth"]["qubits"]) for c in r["program_evidence"])) and
                                   (r["scenario"] != "HARDWARE_ANOMALY" or any(c["component_type"] == r["truth"]["component_type"] and c["component"] == r["truth"]["component"] for c in r["hardware_candidates"]))
                                   for r in records),
        "failure_cases": [r for r in records if r["category_truth"] != r["category_predicted"] or
                          (r["scenario"] == "PROGRAM_FAULT" and not any(c["operation"] == r["truth"]["operation"] and c["qubits"] == list(r["truth"]["qubits"]) for c in r["program_evidence"])) or
                          (r["scenario"] == "HARDWARE_ANOMALY" and not any(c["component_type"] == r["truth"]["component_type"] and c["component"] == r["truth"]["component"] for c in r["hardware_candidates"]))],
    }


def _write_report(records: list[dict[str, Any]], path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    metrics = _rates(records)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["fault_type"] if record["scenario"] == "PROGRAM_FAULT" else record["hardware_type"] if record["scenario"] == "HARDWARE_ANOMALY" else "normal"].append(record)
    lines = ["# Deterministic Pipeline Robustness Evaluation", "", "This report evaluates the executable deterministic diagnosis/localization pipeline via `diagnose_execution`. No ML model or classifier is called; results are not ML accuracy.", "",
        "## Exact evaluation design", "", f"- Cases: **{len(records)}**; shots per distribution: **{SHOTS}**; ideal reference samples are reused for repeated fault/noise variants with identical circuit and seed.",
        f"- Seeds: **{', '.join(map(str, SEEDS))}**.", "- Unseen generated circuit families: GHZ, parity-network, and QFT-like; each is generated from requested width, depth, and seed.",
        f"- Widths: {', '.join(f'{s}={n} qubits' for s,n in WIDTHS)}; depths: {', '.join(map(str, DEPTHS))}.",
        "- Program faults: gate replacement, deletion, insertion, parameter modification, and explicit single-qubit and two-qubit gate faults. Faulty circuits are localized by the existing operation-sequence evidence. All affected qubits are retained as ground truth.",
        "- Hardware scenarios: readout error, gate error, T1 degradation, T2 degradation, and combined degradation; strengths are 0.15, 0.45, and 0.8.",
        "- For each seed/family/size/depth combination: one normal, six program fault, and fifteen hardware anomaly cases.", "",
        "## Actual aggregate results", "", f"- Anomaly detection: **{metrics['anomaly_correct']}/{len(records)}** ({metrics['anomaly_correct']/len(records):.1%}).",
        f"- PROGRAM_FAULT/HARDWARE_ANOMALY/NO_ANOMALY routing: **{metrics['routing_correct']}/{len(records)}** ({metrics['routing_correct']/len(records):.1%}).",
        f"- Program fault localization (exact operation and qubits): **{metrics['program_localized']}/{metrics['program_cases']}** ({metrics['program_localized']/max(1,metrics['program_cases']):.1%}).",
        f"- Hardware localization (affected component present): **{metrics['hardware_localized']}/{metrics['hardware_cases']}** ({metrics['hardware_localized']/max(1,metrics['hardware_cases']):.1%}).",
        f"- End-to-end correct diagnosis and localization: **{metrics['end_to_end_correct']}/{len(records)}** ({metrics['end_to_end_correct']/len(records):.1%}).", "",
        "## Results by scenario type", "", "| Type | Cases | Anomaly correct | Routing correct | Localization correct |", "|---|---:|---:|---:|---:|"]
    for name, subset in sorted(by_type.items()):
        anomaly = sum(r["anomaly_truth"] == r["anomaly_predicted"] for r in subset)
        routing = sum(r["category_truth"] == r["category_predicted"] for r in subset)
        if name in {"normal", "none"}:
            loc = "n/a"
        elif subset[0]["scenario"] == "PROGRAM_FAULT":
            loc = f"{sum(any(c['operation']==r['truth']['operation'] and c['qubits']==list(r['truth']['qubits']) for c in r['program_evidence']) for r in subset)}/{len(subset)}"
        else:
            loc = f"{sum(any(c['component_type']==r['truth']['component_type'] and c['component']==r['truth']['component'] for c in r['hardware_candidates']) for r in subset)}/{len(subset)}"
        lines.append(f"| {name} | {len(subset)} | {anomaly}/{len(subset)} | {routing}/{len(subset)} | {loc} |")
    failure_groups = Counter((r["scenario"], r["fault_type"], r["hardware_type"], r["size"], r["depth"], r["strength"], r["category_predicted"]) for r in metrics["failure_cases"])
    lines += ["", "## Results by size and depth", "", "| Width | Depth | Cases | Anomaly correct | Routing correct | End-to-end correct |", "|---|---:|---:|---:|---:|---:|"]
    for size, n in WIDTHS:
        for depth in DEPTHS:
            subset = [r for r in records if r["size"] == size and r["depth"] == depth]
            anomaly = sum(r["anomaly_truth"] == r["anomaly_predicted"] for r in subset)
            routing = sum(r["category_truth"] == r["category_predicted"] for r in subset)
            e2e = sum(r["category_truth"] == r["category_predicted"] and
                      (r["scenario"] != "PROGRAM_FAULT" or any(c["operation"] == r["truth"]["operation"] and c["qubits"] == list(r["truth"]["qubits"]) for c in r["program_evidence"])) and
                      (r["scenario"] != "HARDWARE_ANOMALY" or any(c["component_type"] == r["truth"]["component_type"] and c["component"] == r["truth"]["component"] for c in r["hardware_candidates"])) for r in subset)
            lines.append(f"| {size} ({n} qubits) | {depth} | {len(subset)} | {anomaly}/{len(subset)} | {routing}/{len(subset)} | {e2e}/{len(subset)} |")
    lines += ["", "## Failure cases", "", f"Total failures against combined expected routing/localization: **{len(metrics['failure_cases'])}**."]
    if failure_groups:
        lines += ["", "| Scenario | Fault | Hardware | Size | Depth | Strength | Predicted category | Failures |", "|---|---|---|---|---:|---:|---|---:|"]
        lines += [f"| {s} | {f} | {h} | {z} | {d} | {st} | {pred} | {count} |" for (s,f,h,z,d,st,pred), count in sorted(failure_groups.items(), key=str)]
        lines += ["", "First 20 individual failure records (seed, truth, predictions and candidate evidence) are included below:", "", "```json", json.dumps(metrics["failure_cases"][:20], indent=2), "```"]
    else:
        lines.append("No failures were observed in these evaluated cases.")
    lines += ["", "## Localization performance", "", "Program success requires the injected operation and exact affected qubit tuple to appear in positive program-fault evidence. Hardware success requires the affected component to appear among relevant hardware evidence candidates. These are top-candidate-inclusion measures, not rank or causal certainty measures.", "",
        "## Limitations", "", "- All observations are generated by the project statevector simulator and its HardwareModel; these are simulated, controlled anomaly conditions.",
        "- Seed variability is measured with three seeds and 128 shots per distribution, across three generated circuit families, three widths, and two depths; this does not establish broad real-circuit generalization.",
        "- Program and hardware scenarios are run as independent axes: hardware degradation is applied only in hardware-anomaly cases; program-fault cases use ideal hardware.",
        "- Hardware T1/T2 values and strengths are controlled calibration inputs. Their simulator effects depend on operation duration and implemented noise semantics.",
        "- No real ML model, external benchmark, or physical quantum hardware is used.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    records = evaluate()
    metrics = _write_report(records)
    print(json.dumps({key: value for key, value in metrics.items() if key != "failure_cases"}, indent=2))
    print(f"Report: {DEFAULT_REPORT}")
    return 0 if not metrics["failure_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
