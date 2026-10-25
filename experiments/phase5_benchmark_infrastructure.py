"""Phase 5 benchmark utilities; this module never runs benchmark experiments.

The 21-template denominator, observable Logistic Regression route, and 0.60
cutoff follow Phase 5.8 and the user's approval. Probability calibration is
optional in the contract and omitted here; LR output probabilities are
uncalibrated and the cutoff is an operational confidence rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import pstdev
from typing import Any, Mapping, Sequence

from quantum_simulator.diagnosis_engine import diagnose_execution
from quantum_simulator.ml_diagnosis import (
    FEATURE_NAMES, MIN_CONFIDENCE, diagnose_with_ml, extract_observable_features,
)


FAMILIES = (
    "F1_GHZ_CHAIN", "F2_TOFFOLI_COMPUTE", "F3_SWAP_ROUTING",
    "F4_PARITY_TREE", "F5_PHASE_LAYER",
)
SUFFIXES = ("N0", "P1", "P2", "R02", "R06", "R15", "R30", "G02", "G06", "G15", "G30", "C15")
PARTITIONS = {
    "development": frozenset(FAMILIES[:2]),
    "calibration": frozenset((FAMILIES[2],)),
    "locked_test": frozenset(FAMILIES[3:]),
}
SEEDS = (20263001, 20263002, 20263003, 20263004, 20263005)
SHOT_COUNTS = (256, 1024, 4096)
# Probability calibration is allowed on F3, but neither the approved reports
# nor the user's steering selects a calibration mapping method.
PROBABILITY_CALIBRATION = "none"

_CIRCUITS = {
    FAMILIES[0]: (("h", 0), ("cx", 0, 1), ("cx", 1, 2), ("cx", 2, 3)),
    FAMILIES[1]: (("x", 0), ("x", 1), ("ccx", 0, 1, 2), ("cx", 2, 3)),
    FAMILIES[2]: (("x", 0), ("x", 2), ("swap", 0, 1), ("swap", 1, 2), ("swap", 2, 3)),
    FAMILIES[3]: (("x", 0), ("h", 1), ("cx", 0, 1), ("cx", 1, 2), ("cx", 1, 3), ("h", 1)),
    FAMILIES[4]: (("h", 0), ("t", 0), ("h", 0), ("cx", 0, 1), ("h", 2), ("t", 2), ("h", 2)),
}
_PROGRAM_MUTATIONS = {
    FAMILIES[0]: ((2, ("cx", 1, 2), None), (2, ("cx", 1, 2), ("cz", 1, 2))),
    FAMILIES[1]: ((2, ("ccx", 0, 1, 2), None), (2, ("ccx", 0, 1, 2), ("ccx", 0, 1, 3))),
    FAMILIES[2]: ((3, ("swap", 1, 2), None), (3, ("swap", 1, 2), ("i", 1))),
    FAMILIES[3]: ((4, ("cx", 1, 3), None), (4, ("cx", 1, 3), ("cz", 1, 3))),
    FAMILIES[4]: ((5, ("t", 2), None), (5, ("t", 2), ("tdg", 2))),
}
_R_TARGET = dict(zip(FAMILIES, (2, 2, 1, 3, 1)))
_G_TARGET = dict(zip(FAMILIES, (("cx", 2, 3), ("ccx", 0, 1, 2), ("swap", 1, 2), ("cx", 1, 3), ("t", 2))))


def _exact_tvd(family: str, suffix: str) -> float:
    if suffix == "N0":
        return 0.0
    if suffix in ("P1", "P2"):
        if family == FAMILIES[2] or (family == FAMILIES[4] and suffix == "P2"):
            return 0.0
        return 1.0 if family == FAMILIES[1] else 0.5
    if suffix.startswith("R"):
        return {"R02": .02, "R06": .06, "R15": .15, "R30": .30}[suffix]
    if suffix.startswith("G"):
        p = {"G02": .02, "G06": .06, "G15": .15, "G30": .30}[suffix]
        return 0.0 if family == FAMILIES[2] else (p if family == FAMILIES[1] else p / 2 if family in (FAMILIES[0], FAMILIES[3]) else ((2-sqrt(2))/4)*p)
    return .15


@dataclass(frozen=True)
class Case:
    case_id: str
    family_id: str
    suffix: str
    expected_circuit: tuple[tuple[Any, ...], ...]
    observed_circuit: tuple[tuple[Any, ...], ...]
    expected_intervention: str
    truth: str | frozenset[str]
    exact_d: str
    exact_tvd: float
    evidence: frozenset[str]
    eligible_metrics: frozenset[str]


def build_ledger() -> tuple[Case, ...]:
    cases: list[Case] = []
    for family in FAMILIES:
        for suffix in SUFFIXES:
            expected = list(_CIRCUITS[family])
            observed = list(expected)
            if suffix in ("P1", "P2", "C15"):
                index, old, replacement = _PROGRAM_MUTATIONS[family][0 if suffix in ("P1", "C15") else 1]
                assert observed[index] == old
                if replacement is None:
                    del observed[index]
                else:
                    observed[index] = replacement
            p = suffix[-2:] if suffix.startswith(("R", "G")) else None
            truth: str | frozenset[str]
            if suffix == "N0": truth = "NORMAL"
            elif suffix in ("P1", "P2"): truth = "PROGRAM_FAULT"
            elif suffix == "C15": truth = frozenset(("PROGRAM_FAULT", "HARDWARE_ANOMALY"))
            else: truth = "HARDWARE_ANOMALY"
            tvd = _exact_tvd(family, suffix)
            evidence = {"expected_circuit", "observed_circuit", "expected_distribution", "observed_counts", "shots", "explicit_calibration"}
            metrics = {"cause_classification"}
            if suffix != "N0": metrics.add("output_anomaly")
            if suffix in ("P1", "P2", "C15"): metrics.add("program_localization")
            if suffix.startswith("R") and suffix != "R02": metrics.add("qubit_localization")
            if suffix.startswith("G") and suffix != "G02": metrics.add("gate_tuple_localization")
            if suffix == "C15": metrics.update(("combined_multilabel_recovery", "combined_ambiguous_disposition", "abstention"))
            if suffix == "N0": metrics.add("specificity_control")
            if tvd == 0:
                metrics.add("specificity_control")
                if suffix != "N0": metrics.add("masked_or_output_equivalent_control")
            if suffix.startswith("R"): intervention = f"readout flip p={p}; target q{_R_TARGET[family]}"
            elif suffix.startswith("G"): intervention = f"gate skip p={p}; target {tuple(_G_TARGET[family])}"
            elif suffix in ("P1", "P2"): intervention = f"program mutation {suffix}"
            elif suffix == "C15": intervention = f"P1 program mutation plus readout flip p=.15 at q{_R_TARGET[family]}"
            else: intervention = "none"
            cases.append(Case(f"{family}-{suffix}", family, suffix, tuple(expected), tuple(observed), intervention, truth, "D+" if tvd > 0 else "D−", tvd, frozenset(evidence), frozenset(metrics)))
    validate_ledger(cases)
    return tuple(cases)


def validate_ledger(cases: Sequence[Case]) -> None:
    expected_ids = {f"{family}-{suffix}" for family in FAMILIES for suffix in SUFFIXES}
    found = [case.case_id for case in cases]
    if len(found) != 60 or len(set(found)) != 60 or set(found) != expected_ids:
        raise ValueError("ledger must contain each of the exact 60 supported cases once")
    if any(c.family_id not in FAMILIES or c.suffix not in SUFFIXES or c.case_id != f"{c.family_id}-{c.suffix}" for c in cases):
        raise ValueError("ledger contains an unsupported or inconsistent case")
    if any(c.exact_d != ("D+" if c.exact_tvd > 0 else "D−") for c in cases):
        raise ValueError("ledger exact-D labels must match reviewed exact TVD")


def validate_family_split(case_families: Sequence[str] | None = None) -> None:
    sets = tuple(PARTITIONS.values())
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]) or set().union(*sets) != set(FAMILIES):
        raise ValueError("family partitions must be disjoint and cover F1–F5")
    if case_families is not None and not set(case_families) <= set(FAMILIES):
        raise ValueError("case contains an unsupported family")


_ALIASES = {"cnot": "cx", "id": "i", "identity": "i", "toffoli": "ccx", "t_dagger": "tdg"}
_ARITY = {"h": 1, "x": 1, "t": 1, "tdg": 1, "i": 1, "cx": 2, "cz": 2, "swap": 2, "ccx": 3}


def _op(op: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(op, Mapping):
        name = str(op.get("name", op.get("op", op.get("gate", "")))).lower()
        raw = op.get("qubits", ())
    elif isinstance(op, (tuple, list)) and op:
        name, raw = str(op[0]).lower(), op[1:]
    else: raise ValueError(f"invalid circuit operation: {op!r}")
    if isinstance(raw, int): raw = (raw,)
    if raw and isinstance(raw[0], (tuple, list)): raw = raw[0]
    return _ALIASES.get(name, name), tuple(raw)


def validate_calibration(profile: Mapping[str, Any] | None, expected_circuit: Sequence[Any], observed_circuit: Sequence[Any], *, case: Case) -> None:
    """Fail closed on explicit q/gate profile fields and both-circuit coverage."""
    if not isinstance(profile, Mapping): raise ValueError("explicit hardware calibration profile is required")
    if set(profile) != {"qubits", "gates", "coupling_map", "edges"}: raise ValueError("profile fields must be exactly qubits, gates, coupling_map, edges")
    if profile["coupling_map"] != [] or profile["edges"] != []: raise ValueError("Phase 5 requires explicit empty coupling_map and edges")
    qubits = profile["qubits"]
    if not isinstance(qubits, Mapping) or set(qubits) != {0, 1, 2, 3} and set(qubits) != {"0", "1", "2", "3"}: raise ValueError("explicit q0-q3 calibration required")
    readout_override = case.suffix.startswith("R") or case.suffix == "C15"
    expected_readout = .15 if case.suffix == "C15" else {"R02": .02, "R06": .06, "R15": .15, "R30": .30}.get(case.suffix, 0.0)
    for q in range(4):
        rec = qubits[q] if q in qubits else qubits[str(q)]
        if not isinstance(rec, Mapping) or set(rec) != {"readout_error", "t1", "t2", "status", "available"}: raise ValueError(f"q{q} calibration is incomplete or has extra fields")
        if type(rec["available"]) is not bool or rec["available"] is not True or rec["status"] != "available": raise ValueError(f"q{q} availability/status invalid")
        for name in ("readout_error", "t1", "t2"):
            v = rec[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not __import__("math").isfinite(v) or v < 0: raise ValueError(f"q{q}.{name} invalid")
        want = expected_readout if readout_override and q == _R_TARGET[case.family_id] else 0.0
        if rec["readout_error"] != want or rec["t1"] != 1e6 or rec["t2"] != 1e6: raise ValueError(f"q{q} calibration does not match explicit baseline/allowed override")
    required: set[tuple[str, tuple[int, ...]]] = set()
    for circuit in (expected_circuit, observed_circuit):
        for raw in circuit:
            name, qubits_tuple = _op(raw)
            if name in ("measure", "barrier"): continue
            if name not in _ARITY or len(qubits_tuple) != _ARITY[name] or any(type(q) is not int or not 0 <= q < 4 for q in qubits_tuple): raise ValueError(f"invalid gate operation {raw!r}")
            required.add((name, qubits_tuple))
    entries = profile["gates"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)): raise ValueError("gates must be a list of explicit records")
    actual: dict[tuple[str, tuple[int, ...]], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"gate", "qubits", "error", "duration", "supported"}: raise ValueError("gate calibration record must explicitly contain gate, qubits, error, duration, supported")
        key = (_ALIASES.get(str(entry["gate"]).lower(), str(entry["gate"]).lower()), tuple(entry["qubits"]))
        if key in actual: raise ValueError("duplicate canonical gate calibration")
        actual[key] = entry
    if set(actual) != required: raise ValueError("gate profile must exactly cover expected/observed canonical gate tuple union")
    for key, entry in actual.items():
        err, dur = entry["error"], entry["duration"]
        if type(entry["supported"]) is not bool or entry["supported"] is not True or isinstance(err, bool) or not isinstance(err, (int,float)) or not 0 <= err <= 1 or isinstance(dur, bool) or not isinstance(dur,(int,float)) or not __import__("math").isfinite(dur) or dur < 0: raise ValueError(f"invalid calibration values for {key}")
        is_fault = case.suffix.startswith("G") and key == (_G_TARGET[case.family_id][0], tuple(_G_TARGET[case.family_id][1:]))
        want = {"G02": .02, "G06": .06, "G15": .15, "G30": .30}.get(case.suffix, 0.0) if is_fault else 0.0
        if err != want or dur != 0.0: raise ValueError(f"unexpected gate calibration/override for {key}")


_FORBIDDEN_KEYS = frozenset(("label", "truth", "cause", "intervention", "family", "family_id", "split", "seed", "case_id", "target", "strength", "locked_test", "prediction", "result"))


def observable_features(expected: Mapping[str, float], observed: Mapping[str, float]) -> dict[str, float]:
    """Return the existing exact five observable features in frozen order."""
    features = extract_observable_features(expected, observed)
    if tuple(features) != FEATURE_NAMES: raise ValueError("observable feature order differs from frozen schema")
    return features


def build_inference_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"expected_distribution", "observed_distribution", "shots", "expected_circuit", "observed_circuit", "hardware_model", "threshold"}
    if not isinstance(record, Mapping) or set(record) - allowed: raise ValueError("inference payload contains forbidden or unsupported fields")
    required = {"expected_distribution", "observed_distribution", "shots", "expected_circuit", "observed_circuit", "hardware_model", "threshold"}
    if set(record) != required: raise ValueError("inference payload is missing required evidence")
    if type(record["shots"]) is not int or record["shots"] not in SHOT_COUNTS or record["threshold"] != .05: raise ValueError("shots or fixed threshold violate Phase 5 protocol")
    return dict(record)


def anomaly_observation(tvd: float, shots: int) -> dict[str, Any]:
    if type(shots) is not int or shots not in SHOT_COUNTS: raise ValueError("unsupported shot count")
    guard = max(.05, 1 / sqrt(shots))
    return {"tvd": float(tvd), "threshold": .05, "finite_shot_guard": guard, "detected": float(tvd) > guard}


def score_observation(case: Case, *, anomaly_detected: bool | None = None, predicted_cause: str | Sequence[str] | None = None, predicted_targets: Sequence[str] = (), disposition: str | None = None) -> dict[str, Any]:
    """Score one observation; acceptance aggregation is deliberately gated."""
    result: dict[str, Any] = {"truth": case.truth, "exact_d": case.exact_d, "anomaly_prediction": anomaly_detected, "output_detection_eligible": True, "output_positive_truth": case.exact_d == "D+"}
    if case.suffix == "C15":
        predicted = frozenset((predicted_cause,)) if isinstance(predicted_cause, str) else frozenset(predicted_cause or ())
        result.update(combined_recovery=(predicted == case.truth), categorical_ambiguous=(disposition == "AMBIGUOUS"), single_cause_prediction=predicted in (frozenset(("PROGRAM_FAULT",)), frozenset(("HARDWARE_ANOMALY",))))
    elif isinstance(case.truth, str):
        result["cause_correct"] = predicted_cause == case.truth
    result["eligible_localization_metrics"] = sorted(x for x in case.eligible_metrics if x.endswith("localization") or x == "gate_tuple_localization")
    result["predicted_targets"] = tuple(predicted_targets)
    return result


_SENSITIVITY_CASES = frozenset(
    [f"{family}-{suffix}" for family in FAMILIES for suffix in ("R15", "R30", "C15")]
    + [f"{family}-{suffix}" for family in (FAMILIES[0], FAMILIES[1], FAMILIES[3]) for suffix in ("G15", "G30")]
)

# Owner decision recorded in the Phase 5.8 decision memo addendum (2026-10-13):
# formal specificity is N0 plus the three masked D− controls. F3 G cases remain
# a separate diagnostic stratum and never enter this denominator.
_SPECIFICITY_CASES = frozenset(
    [f"{family}-N0" for family in FAMILIES]
    + [f"{FAMILIES[2]}-P1", f"{FAMILIES[2]}-P2", f"{FAMILIES[4]}-P2"]
)
_F3_GATE_DIAGNOSTIC_CASES = frozenset(f"{FAMILIES[2]}-{suffix}" for suffix in ("G02", "G06", "G15", "G30"))


def aggregate_output_detection(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the approved 21-template sensitivity set without pseudo-replication.

    Input is the full 60-case x five-seed x three-shot ledger of sampled
    decisions. Template membership comes only from the frozen case ID set and
    exact-D truth; sampled outcomes are never used to establish eligibility.
    """
    cases = {case.case_id: case for case in build_ledger()}
    expected_keys = {(case_id, seed, shots) for case_id in cases for seed in SEEDS for shots in SHOT_COUNTS}
    rows: dict[tuple[str, int, int], bool] = {}
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "seed", "shots", "anomaly_detected"}:
            raise ValueError("each observation must contain only case_id, seed, shots, anomaly_detected")
        case_id, seed, shots = row["case_id"], row["seed"], row["shots"]
        key = (case_id, seed, shots)
        if key not in expected_keys:
            raise ValueError("unsupported case, seed, or shot configuration")
        if key in rows:
            raise ValueError("duplicate configuration; repeats are not independent templates")
        if type(row["anomaly_detected"]) is not bool:
            raise ValueError("anomaly_detected must be a boolean sampled decision")
        rows[key] = row["anomaly_detected"]
    if set(rows) != expected_keys:
        raise ValueError("all 900 frozen case/seed/shot configurations are required")

    negative_ids = _SPECIFICITY_CASES
    by_shot: dict[int, dict[str, Any]] = {}
    for shots in SHOT_COUNTS:
        positive = [(case_id, seed, rows[(case_id, seed, shots)]) for case_id in _SENSITIVITY_CASES for seed in SEEDS]
        negative = [(case_id, seed, rows[(case_id, seed, shots)]) for case_id in negative_ids for seed in SEEDS]
        sensitivity = sum(detected for _, _, detected in positive) / len(positive)
        specificity = sum(not detected for _, _, detected in negative) / len(negative)
        per_family = {}
        for family in FAMILIES:
            family_positive = [v for case_id, _, v in positive if case_id.startswith(family + "-")]
            per_family[family] = {"detected": sum(family_positive), "denominator": len(family_positive),
                                  "sensitivity": sum(family_positive) / len(family_positive)}
        per_seed = [sum(detected for case_id, seed, detected in positive if seed == s) / len(_SENSITIVITY_CASES) for s in SEEDS]
        by_shot[shots] = {
            "sensitivity": sensitivity, "sensitivity_numerator": sum(d for _, _, d in positive),
            "sensitivity_denominator": len(positive), "sensitivity_templates": len(_SENSITIVITY_CASES),
            "specificity": specificity, "specificity_numerator": sum(not d for _, _, d in negative),
            "specificity_denominator": len(negative), "negative_templates": len(negative_ids),
            "sensitivity_acceptance": sensitivity >= .80,
            "specificity_acceptance": specificity >= .90,
            "per_family_sensitivity": per_family,
            "per_seed_sensitivity_mean": sum(per_seed) / len(per_seed),
            "per_seed_sensitivity_population_sd": pstdev(per_seed),
        }
    return {
        "sensitivity_templates": 21,
        "observations_per_shot_level": 105,
        "total_sensitivity_observations": 315,
        "specificity_templates": len(negative_ids),
        "specificity_membership": "owner decision 2026-10-13: N0 plus masked D− controls; F3 G cases reported separately",
        "reserved_family_f3_gate_diagnostic": {
            shots: {
                case_id: {
                    "false_positives": sum(rows[(case_id, seed, shots)] for seed in SEEDS),
                    "denominator": len(SEEDS),
                    "sampled_detection_rate": sum(rows[(case_id, seed, shots)] for seed in SEEDS) / len(SEEDS),
                }
                for case_id in sorted(_F3_GATE_DIAGNOSTIC_CASES)
            }
            for shots in SHOT_COUNTS
        },
        "per_shot": by_shot,
        "pooled_sensitivity_descriptive_only": sum(v["sensitivity_numerator"] for v in by_shot.values()) / 315,
        "repeated_configurations_are_independent_templates": False,
    }


def classifier_inference(
    expected_distribution: Mapping[str, float], observed_distribution: Mapping[str, float], *,
    expected_circuit: Sequence[Any], observed_circuit: Sequence[Any], hardware_model: Any,
    shots: int, classifier: Any, combined_truth: bool = False, precomputed_trace: Any = None,
) -> dict[str, Any]:
    """Run the fixed-cutoff observable LR route and separate combined adjudication."""
    ml = diagnose_with_ml(
        expected_distribution, observed_distribution, expected_circuit=expected_circuit,
        observed_circuit=observed_circuit, hardware_model=hardware_model, threshold=.05,
        shots=shots, precomputed_trace=precomputed_trace, classifier=classifier,
        min_confidence=MIN_CONFIDENCE,
    )
    disposition = ml.final_diagnosis.value
    if combined_truth and ml.confidence >= MIN_CONFIDENCE:
        # The classifier's single-cause output remains separately recorded.
        # The existing evidence adjudicator determines C15's categorical result.
        combined = diagnose_execution(
            expected_distribution, observed_distribution, expected_circuit=expected_circuit,
            observed_circuit=observed_circuit, hardware_model=hardware_model,
            threshold=.05, shots=shots, precomputed_trace=precomputed_trace,
        )
        disposition = combined.category.value
    return {
        "predicted_class": ml.predicted_class,
        "class_probabilities": dict(ml.class_probabilities),
        "confidence": ml.confidence,
        "features": dict(ml.ml_features),
        "confidence_cutoff": MIN_CONFIDENCE,
        "probabilities_calibrated": False,
        "confidence_policy": "operational 0.60 cutoff on uncalibrated Logistic Regression probabilities; not a guarantee of 60% empirical correctness",
        "disposition": disposition,
    }



