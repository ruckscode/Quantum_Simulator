"""Structured, result-specific explanations for the diagnosis pipeline.

The payload describes the calculations the engine actually performed. It is
intended for a UI or report and contains no credentials or circuit source text.
"""

from __future__ import annotations

from typing import Any

from .diagnosis_engine import DiagnosisResult


def explain_diagnosis(
    result: DiagnosisResult,
    *,
    n_qubits: int | None = None,
    shots: int | None = None,
) -> list[dict[str, Any]]:
    """Return stage-by-stage equations, observed values, and decision meaning.

    This does not recalculate or alter a diagnosis. Fields absent from a result
    are reported as unavailable rather than inferred.
    """
    if not isinstance(result, DiagnosisResult):
        raise TypeError("result must be a DiagnosisResult")

    steps: list[dict[str, Any]] = []

    def add(stage: str, algorithm: str, equation: str, values: dict[str, Any], meaning: str) -> None:
        steps.append({
            "stage": stage,
            "algorithm": algorithm,
            "equation": equation,
            "values": values,
            "meaning": meaning,
        })

    add(
        "ideal state evolution and measurement model",
        "complex statevector with Born-rule measurement and finite-shot sampling",
        "|psi_(t+1)> = U_t |psi_t>;  p(x)=|<x|psi>|^2;  c ~ Multinomial(N,p)",
        {"qubits": n_qubits, "statevector_dimension": 2 ** n_qubits if n_qubits is not None else None,
         "shots": shots, "expected_outcomes": result.expected_counts,
         "observed_outcomes": result.observed_counts},
        "The statevector engine models ideal unitary gates; counts are sampled outcomes. "
        "A real backend result is observed hardware data, while a HardwareModel run is synthetic.",
    )

    stats = result.statistics_result
    if stats is None:
        add("distribution comparison", "not run", "TVD = 1/2 sum_x |p(x)-q(x)|", {},
            "Expected or observed distribution evidence was unavailable.")
    else:
        values = {
            "total_variation_distance": stats.total_variation_distance,
            "hellinger_distance": stats.hellinger_distance,
            "jensen_shannon_divergence_bits": stats.js_divergence,
            "configured_effect_threshold": stats.threshold,
            "finite_shot_guard": stats.finite_shot_guard,
            "effective_threshold": max(stats.threshold, stats.finite_shot_guard),
            "shots": stats.shots,
            "anomaly_detected": stats.anomaly_detected,
        }
        if hasattr(stats, "p_value"):
            values.update({"statistical_test": getattr(stats, "method", "count-aware test"),
                           "p_value": stats.p_value})
            equation = "TVD = 1/2 sum_x |p(x)-q(x)|; detect iff p_value <= alpha AND TVD > effect_threshold"
            meaning = "Count-aware hypothesis testing separates statistical significance from practical effect size."
        else:
            equation = "TVD = 1/2 sum_x |p(x)-q(x)|; detect iff TVD > max(effect_threshold, 1/sqrt(N))"
            meaning = "Probability-only comparison applies a finite-shot tolerance; the other distances are descriptive metrics."
        add("distribution comparison", "TVD with finite-shot guard or count-aware bootstrap/permutation",
            equation, values, meaning)

    if result.program_result is not None:
        candidates = result.program_evidence_candidates
        add(
            "program localization",
            "operation-signature alignment / edit-distance candidates plus output-impact counterfactual",
            "D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+[a_i != b_j])",
            {"candidate_count": len(candidates),
             "candidates": [{"index": getattr(c, "index", None),
                             "expected": str(getattr(c, "expected_signature", None)),
                             "observed": str(getattr(c, "actual_signature", None)),
                             "score": getattr(c, "score", None),
                             "reason": getattr(c, "reason", None)} for c in candidates],
             "behavioral_candidate_count": len(result.program_behavioral_candidates),
             "trace_only_candidate_count": len(result.trace_deviation_candidates)},
            "A source mismatch is only classified as a program fault when it can affect measured outputs and execution evidence supports an anomaly.",
        )
    else:
        add("program localization", "not run", "operation sequence alignment", {},
            "Expected/observed source-operation evidence was unavailable.")

    if result.hardware_result is not None:
        candidates = result.hardware_evidence_candidates
        freshness = result.calibration_freshness or {}
        add(
            "hardware localization",
            "circuit-relevant calibration rule scoring",
            "S_q=1.5[I_unavailable]+4e_ro[I(e_ro>0.05)]+max(0,(100-T1)/100)[I(T1<100)]+max(0,(100-T2)/100)[I(T2<100)]+0.5min(n_q_ops,3); S_g=1.5[I_unsupported]+10e_g[I(e_g>0.05)]; S_edge=1.5[I_unsupported]+8e_edge[I(e_edge>0.05)]+0.75min(n_edge_ops,3)",
            {"candidate_count": len(candidates),
             "candidates": [{"component_type": getattr(c, "component_type", None),
                             "component": str(getattr(c, "component", None)),
                             "score": getattr(c, "score", None),
                             "reason": getattr(c, "reason", None),
                             "evidence": list(getattr(c, "evidence", []))} for c in candidates],
             "calibration_freshness": freshness,
             "missing_evidence": list(result.missing_evidence)},
            "Scores are heuristic candidate ranks, not probabilities. Only relevant calibration evidence can support hardware attribution. Synthetic calibration is labeled as a fixture and does not establish physical-device performance.",
        )
    else:
        add("hardware localization", "not run", "candidate = f(calibration, circuit)",
            {"missing_evidence": list(result.missing_evidence)},
            "No usable calibration model was supplied; hardware cause attribution is limited.")

    if result.ml_predicted_class is not None:
        add(
            "optional classical ML advisor",
            "standardized observable features followed by multiclass Logistic Regression softmax",
            "z=(x-mu)/s;  P(y=k|x)=exp(w_k^T z+b_k)/sum_j exp(w_j^T z+b_j)",
            {"class": result.ml_predicted_class, "uncalibrated_confidence": result.ml_confidence,
             "class_probabilities": result.ml_class_probabilities,
             "features": result.ml_features, "deployment_validated": False},
            "This candidate is never loaded implicitly. Its probabilities are uncalibrated and cannot override conflicting independent evidence.",
        )
    else:
        add("optional classical ML advisor", "not invoked", "no model call unless classifier= is explicitly supplied", {},
            "The current diagnosis is rule/statistics/localization based; benchmark classifiers are not silently loaded.")

    add(
        "final evidence fusion",
        "conservative evidence rules with abstention",
        "PROGRAM_FAULT only with behavioral source evidence; HARDWARE_ANOMALY only with relevant calibration evidence; conflicts or gaps => AMBIGUOUS/INSUFFICIENT",
        {"category": result.category.value, "anomaly_detected": result.anomaly_detected,
         "evidence": list(result.evidence), "missing_evidence": list(result.missing_evidence),
         "summary": result.summary},
        "A diagnosis category is a conclusion from the available evidence, not a universal accuracy guarantee.",
    )
    return steps


__all__ = ["explain_diagnosis"]
