"""Optional local Streamlit interface for circuit simulation and diagnosis."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from quantum_simulator import HardwareModel, diagnose_circuit, explain_diagnosis


DEFAULT_BELL = json.dumps(
    [["h", 0], ["cx", 0, 1], ["measure", [0, 1]]], indent=2,
)


def _parse_operations(raw: str, name: str) -> list[list[Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON (line {exc.lineno}, column {exc.colno}).") from None
    if not isinstance(value, list) or any(not isinstance(item, list) or not item for item in value):
        raise ValueError(f"{name} must be a JSON list of operation lists, such as [[\"h\", 0], [\"measure\", [0]]].")
    if any(not isinstance(item[0], str) for item in value):
        raise ValueError(f"Every operation in {name} must start with a gate name string.")
    return value


def _two_qubit_edges(operations: list[list[Any]]) -> list[tuple[int, int]]:
    two_qubit = {"cx", "cnot", "cz", "swap", "ch", "cp", "crx", "cry", "crz", "rxx", "rzz"}
    edges = set()
    for operation in operations:
        if operation[0].lower() in two_qubit and len(operation) >= 3:
            edges.add(tuple(sorted((int(operation[1]), int(operation[2])))))
    return sorted(edges)


def _gate_qubits(operation: list[Any]) -> tuple[str, tuple[int, ...]] | None:
    name = operation[0].lower()
    if name in {"measure", "reset", "barrier"}:
        return None
    arity = 3 if name in {"ccx", "rccx", "toffoli", "cswap", "fredkin"} else (
        2 if name in {"cx", "cnot", "cz", "swap", "ch", "cp", "crx", "cry", "crz", "rxx", "rzz"}
        else 1
    )
    if len(operation) < arity + 1 or any(not isinstance(value, int) or isinstance(value, bool) for value in operation[1:arity + 1]):
        raise ValueError(f"Operation {name!r} must provide {arity} integer qubit index/indices before gate parameters.")
    normalized = {"cnot": "cx", "id": "i", "identity": "i"}.get(name, name)
    return normalized, tuple(operation[1:arity + 1])


def _required_qubits(*programs: list[list[Any]]) -> int:
    used: list[int] = []
    for program in programs:
        for operation in program:
            name = operation[0].lower()
            if name == "measure" and len(operation) > 1:
                values = operation[1] if isinstance(operation[1], list) else [operation[1]]
                used.extend(int(value) for value in values if isinstance(value, int) and not isinstance(value, bool))
            elif name == "barrier":
                used.extend(int(value) for value in operation[1:] if isinstance(value, int) and not isinstance(value, bool))
            else:
                gate = _gate_qubits(operation)
                if gate is not None:
                    used.extend(gate[1])
    if any(qubit < 0 for qubit in used):
        raise ValueError("Qubit indices must be non-negative.")
    return max(used, default=-1) + 1


def _synthetic_model(n_qubits: int, operations: list[list[Any]], strength: float) -> HardwareModel:
    edges = _two_qubit_edges(operations)
    model = HardwareModel(
        n_qubits,
        coupling_map=edges,
        backend_name="synthetic-stress-fixture",
        calibration_timestamp="synthetic-config",
        calibration_timestamp_provenance="synthetic",
        default_t1=100.0,
        default_t2=80.0,
        default_readout_error=strength,
        default_gate_error=strength,
        default_gate_duration=0.5,
    )
    for left, right in edges:
        model.add_edge(left, right, error=strength, duration=0.5, supported=True)
    for operation in operations:
        gate = _gate_qubits(operation)
        if gate is not None:
            name, qubits = gate
            model.add_gate_calibration(name, qubits=qubits, error=strength, duration=0.5, supported=True)
    return model


def _clear_credentials() -> None:
    st.session_state.pop("ibm_api_key_session_only", None)
    st.session_state.pop("ibm_instance_session_only", None)


st.set_page_config(page_title="Quantum Simulator Diagnosis", page_icon="⚛️", layout="wide")
st.title("Quantum circuit simulator and diagnosis")
st.caption("Local statevector simulation, source comparison, synthetic hardware stress, or explicitly submitted IBM hardware execution.")
st.info("The default diagnosis is statistical and evidence-based. Classical/QML benchmark models are not silently called because their training targets do not match this diagnosis task.")

mode = st.selectbox(
    "Execution scenario",
    ["Normal / ideal", "Program-fault comparison", "Synthetic hardware anomaly", "Mixed program + synthetic hardware", "Real IBM hardware"],
)

left, right = st.columns(2)
with left:
    reference_text = st.text_area("Expected/reference circuit (JSON operations)", value=DEFAULT_BELL, height=220)
with right:
    default_submitted = DEFAULT_BELL
    if mode in {"Program-fault comparison", "Mixed program + synthetic hardware"}:
        default_submitted = json.dumps([["h", 0], ["measure", [0, 1]]], indent=2)
    submitted_text = st.text_area("Submitted circuit (JSON operations)", value=default_submitted, height=220)

controls = st.columns(3)
with controls[0]:
    n_qubits = st.number_input("Number of qubits", min_value=1, max_value=16, value=2, step=1)
with controls[1]:
    shots = st.number_input("Shots", min_value=100, max_value=10_000, value=2048, step=256)
with controls[2]:
    seed = st.number_input("Reproducible seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

strength = 0.15
if mode in {"Synthetic hardware anomaly", "Mixed program + synthetic hardware"}:
    strength = st.slider("Synthetic stress probability (gate/readout)", 0.05, 0.40, 0.15, 0.01)
    st.warning("Synthetic fixture only: these stress values are user-selected, not copied from an IBM backend and not a hardware accuracy claim.")

token = ""
backend_name = ""
instance = ""
if mode == "Real IBM hardware":
    st.warning("Real mode can submit a billable/quota-consuming job. The key is kept in this app session memory only; do not use a remotely hosted app you do not control.")
    token = st.text_input("IBM Quantum API key", type="password", key="ibm_api_key_session_only")
    backend_name = st.text_input("IBM backend name", placeholder="e.g. a backend available to your account")
    instance = st.text_input("IBM instance CRN (if your account requires it)", type="password", key="ibm_instance_session_only")
    st.button("Clear IBM key and instance", on_click=_clear_credentials)
    st.caption("The app does not write the key to a file, environment variable, URL, result, or app-owned log. It is not cleared by normal reruns; use the clear button or close the session.")
    submit_real_job = st.checkbox("I understand that running this mode submits a real IBM job and may use account quota.")
else:
    submit_real_job = False

run = st.button("Run simulator", type="primary")
if run:
    try:
        reference = _parse_operations(reference_text, "Reference circuit")
        submitted = _parse_operations(submitted_text, "Submitted circuit")
        if mode == "Normal / ideal":
            submitted = reference
        if mode == "Synthetic hardware anomaly":
            submitted = reference
        required_n_qubits = max(1, _required_qubits(reference, submitted))

        if mode == "Real IBM hardware":
            if not submit_real_job:
                st.error("Confirm real job submission before running. No job was submitted.")
            elif not token or not backend_name:
                st.error("Enter an IBM API key and backend name to run real hardware. No synthetic fallback is performed.")
            else:
                from ibm_validation import execute_on_ibm_backend

                live = execute_on_ibm_backend(
                    submitted,
                    reference_circuit=reference,
                    backend_name=backend_name,
                    token=token,
                    instance=instance or None,
                    shots=int(shots),
                )
                payload = live.to_dict()
                st.subheader("Real IBM execution")
                st.write({key: payload.get(key) for key in (
                    "overall_status", "execution_status", "execution_origin", "real_hardware_executed",
                    "backend", "job_id", "shots", "observed_counts", "diagnosis", "errors",
                )})
                with st.expander("Real-run equations and evidence values", expanded=True):
                    anomaly = payload.get("anomaly") or {}
                    st.json({
                        "expected_probabilities": "p(x)=|<x|psi_reference>|^2 from the ideal reference circuit",
                        "observed_probabilities": "q(x)=c_x/N from the real backend counts",
                        "comparison": {
                            "equation": "TVD=1/2 sum_x |p(x)-q(x)|; count-aware anomaly requires bootstrap p<=alpha and TVD>effect threshold",
                            "test": anomaly.get("test"),
                            "TVD": anomaly.get("total_variation_distance"),
                            "p_value": anomaly.get("p_value"),
                            "effect_threshold": anomaly.get("threshold"),
                            "finite_shot_guard": anomaly.get("finite_shot_guard"),
                            "detected": anomaly.get("detected"),
                        },
                        "program_localization_equation": "D(i,j)=min(delete, insert, substitute); candidate output impact is checked counterfactually",
                        "hardware_candidate_scoring": "S_q/S_gate/S_edge are heuristic ranks; equations are documented in docs/algorithm_reference.md",
                        "program_candidates": payload.get("program_candidates"),
                        "hardware_candidates": payload.get("hardware_candidates"),
                        "calibration_assessment": payload.get("calibration_assessment"),
                    })
        else:
            model = None
            if mode in {"Synthetic hardware anomaly", "Mixed program + synthetic hardware"}:
                if int(n_qubits) < required_n_qubits:
                    raise ValueError(f"Number of qubits must be at least {required_n_qubits} for these circuits.")
                model = _synthetic_model(int(n_qubits), submitted, float(strength))
            diagnosis = diagnose_circuit(
                submitted,
                reference_circuit=None if mode == "Normal / ideal" else reference,
                hardware_model=model,
                shots=int(shots),
                seed=int(seed),
            )
            st.subheader(f"Diagnosis: {diagnosis.category.value}")
            st.write(diagnosis.summary)
            st.write({
                "observed_counts": diagnosis.observed_counts,
                "evidence": diagnosis.evidence,
                "missing_evidence": diagnosis.missing_evidence,
            })
            with st.expander("Mathematical steps and run-specific values", expanded=True):
                st.json(explain_diagnosis(diagnosis, n_qubits=required_n_qubits, shots=int(shots)))
    except Exception as exc:  # Never show a traceback that could include a credential.
        safe_message = str(exc)
        for secret in (token, instance):
            if secret:
                safe_message = safe_message.replace(secret, "[REDACTED]")
        st.error(f"Could not complete this run ({type(exc).__name__}). {safe_message}")
