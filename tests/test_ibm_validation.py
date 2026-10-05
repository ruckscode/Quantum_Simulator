from types import SimpleNamespace
import importlib.util
import sys

import pytest

from quantum_simulator.hardware_model import HardwareModel
from ibm_validation import (
    build_circuit_plan,
    build_ibm_validation_report,
    capture_backend_calibration,
    check_ibm_access_status,
    convert_to_qiskit_circuit,
    counts_to_probabilities,
    discover_ibm_backends,
    execute_on_ibm_backend,
    run_offline_validation,
)


class BackendFixture:
    name = "offline-fixture-backend"
    num_qubits = 2
    coupling_map = [(0, 1)]

    def properties(self):
        return SimpleNamespace(
            qubits=[
                [SimpleNamespace(name="T1", value=0.00012, unit="s"), SimpleNamespace(name="T2", value=0.00008, unit="s"), SimpleNamespace(name="readout_error", value=0.02, unit=None)],
                [SimpleNamespace(name="T1", value=0.00010, unit="s"), SimpleNamespace(name="T2", value=0.00007, unit="s"), SimpleNamespace(name="readout_error", value=0.03, unit=None)],
            ],
            gates=[
                SimpleNamespace(name="cx", gate="cx", qubits=[0, 1], parameters=[SimpleNamespace(name="gate_error", value=0.01, unit=None), SimpleNamespace(name="gate_length", value=2e-7, unit="s")]),
                SimpleNamespace(name="sx", gate="sx", qubits=[0], parameters=[SimpleNamespace(name="gate_error", value=0.002, unit=None), SimpleNamespace(name="gate_length", value=3e-8, unit="s")]),
            ],
        )

    def status(self):
        return SimpleNamespace(operational=True, status_msg="active", pending_jobs=1)


class BackendServiceFixture:
    def backends(self):
        return [BackendFixture()]


def test_backend_discovery_parses_injected_offline_fixture_without_claiming_live_discovery():
    result = discover_ibm_backends(service=BackendServiceFixture(), token="fixture-token")

    assert result["status"] == "PARSED_INJECTED_SERVICE"
    assert result["discovery_source"] == "INJECTED_SERVICE_FIXTURE"
    assert result["backends"] == [{"name": "offline-fixture-backend", "num_qubits": 2, "operational": True}]
    assert result["access"]["status"] == "ACCESS_CONFIGURED"


def test_discovery_uses_explicit_runtime_credentials(monkeypatch):
    import ibm_validation

    calls = {}

    class RuntimeServiceFixture:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def backends(self):
            return [BackendFixture()]

    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(QiskitRuntimeService=RuntimeServiceFixture))
    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))

    result = discover_ibm_backends(token="user-token", instance="user/instance", channel="ibm_quantum_platform")

    assert calls == {"channel": "ibm_quantum_platform", "token": "user-token", "instance": "user/instance"}
    assert result["status"] == "DISCOVERED"


@pytest.mark.parametrize("token", ["", "   ", None])
def test_discovery_blocks_missing_token_without_constructing_runtime(monkeypatch, token):
    import ibm_validation

    def forbidden(**kwargs):
        pytest.fail("Runtime service must not be constructed without an explicit token")

    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(QiskitRuntimeService=forbidden))
    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda value: ibm_validation.IBMAccessStatus("BLOCKED", True, True, False, None, "token required"))

    result = discover_ibm_backends(token=token)

    assert result["status"] == "BLOCKED"


def test_execute_passes_explicit_credentials_without_saved_account_fallback(monkeypatch):
    import ibm_validation

    calls = {}

    class RuntimeServiceFixture:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def backend(self, name):
            calls["backend_name"] = name
            raise RuntimeError("stop after authenticated backend selection")

    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(QiskitRuntimeService=RuntimeServiceFixture))
    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))
    monkeypatch.setattr(ibm_validation, "build_circuit_plan", lambda circuit: ibm_validation.CircuitPlan(1, []))
    monkeypatch.setattr(ibm_validation, "ideal_distribution", lambda circuit, width: {"0": 1.0})
    monkeypatch.setattr(ibm_validation, "convert_to_qiskit_circuit", lambda circuit: ibm_validation.CircuitConversionResult("CONVERTED", object(), ibm_validation.CircuitPlan(1, [])))
    monkeypatch.setattr(ibm_validation, "check_ibm_access_status", lambda: pytest.fail("must not inspect saved account credentials"))

    result = execute_on_ibm_backend([("x", 0)], backend_name="user-backend", token="user-token", instance="user/instance")

    assert calls == {"channel": "ibm_quantum_platform", "token": "user-token", "instance": "user/instance", "backend_name": "user-backend"}
    assert result.overall_status == "FAILED"
    assert "Backend selection failed" in result.errors[0]


def test_execute_blocks_empty_token_without_runtime_service(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("BLOCKED", True, True, False, None, "token required"))
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", SimpleNamespace(QiskitRuntimeService=lambda **kwargs: pytest.fail("must not authenticate")))

    result = execute_on_ibm_backend([("x", 0)], backend_name="user-backend", token="")

    assert result.overall_status == "BLOCKED"
    assert result.real_hardware_executed is False


def test_calibration_conversion_includes_backend_qubit_gate_and_edge_data():
    capture = capture_backend_calibration(BackendFixture())

    assert capture.backend_name == "offline-fixture-backend"
    assert capture.n_qubits == 2
    assert capture.coupling_map == [(0, 1)]
    assert capture.backend_status["operational"] is True
    assert capture.qubits[0]["readout_error"] == 0.02
    assert capture.hardware_model.get_qubit_properties(0).t1 == pytest.approx(120.0)
    assert capture.hardware_model.get_qubit_properties(0).t2 == pytest.approx(80.0)
    assert capture.hardware_model.get_gate_calibration("cx", (0, 1))[0].error == 0.01
    assert capture.hardware_model.get_coupling_map() == [(0, 1)]


def test_missing_backend_calibration_fields_are_explicit_nulls():
    backend = SimpleNamespace(name="partial-fixture", num_qubits=1, coupling_map=None, properties=lambda: None, status=lambda: None)

    capture = capture_backend_calibration(backend)

    assert capture.qubits[0]["t1"] is None
    assert capture.qubits[0]["t2"] is None
    assert capture.qubits[0]["readout_error"] is None
    assert capture.coupling_map is None
    assert "qubits[0].t1" in capture.unavailable_fields
    assert "gate_calibrations" in capture.unavailable_fields


def test_malformed_backend_data_returns_structured_errors():
    backend = SimpleNamespace(name="bad-fixture", num_qubits=0, coupling_map=[(0, 0)], properties=lambda: None, status=lambda: None)

    capture = capture_backend_calibration(backend)

    assert capture.retrieval_status == "PARTIAL"
    assert any("Malformed backend qubit count" in error or "invalid coupling edge" in error for error in capture.errors)


def test_malformed_gate_calibration_is_recorded_without_raising():
    backend = SimpleNamespace(
        name="bad-gate-fixture",
        num_qubits=1,
        coupling_map=[],
        properties=lambda: SimpleNamespace(qubits=[[]], gates=[SimpleNamespace(gate="cx", qubits=[3], parameters=[])]),
        status=lambda: None,
    )

    capture = capture_backend_calibration(backend)

    assert any("Malformed gate calibration" in error for error in capture.errors)
    assert capture.gates == []


def test_circuit_conversion_plan_supports_project_gate_surface():
    circuit = [
        ("i", 0), ("x", 0), ("y", 0), ("z", 0), ("h", 0), ("s", 0), ("sdg", 0),
        ("t", 0), ("tdg", 0), ("sx", 0), ("sxdg", 0),
        ("rx", 0, 0.1), ("ry", 0, 0.2), ("rz", 0, 0.3), ("p", 0, 0.4),
        ("u1", 0, 0.5), ("u2", 0, 0.5, 0.6), ("u3", 0, 0.5, 0.6, 0.7),
        ("cx", 0, 1), ("cnot", 0, 1), ("cz", 0, 1), ("swap", 0, 1), ("ch", 0, 1),
        ("cp", 0, 1, 0.1), ("crx", 0, 1, 0.1), ("cry", 0, 1, 0.1), ("crz", 0, 1, 0.1),
        ("ccx", 0, 1, 2), ("toffoli", 0, 1, 2), ("cswap", 0, 1, 2),
        ("measure", 0), ("reset", 0), ("barrier", 0, 1, 2),
    ]

    plan = build_circuit_plan(circuit)

    assert plan.supported
    assert plan.num_qubits == 3
    assert len(plan.operations) == len(circuit)


def test_unsupported_gate_is_explicit_and_never_substituted():
    result = convert_to_qiskit_circuit([("unknown_gate", 0)])

    assert result.status == "UNSUPPORTED"
    assert result.circuit is None
    assert result.plan.unsupported_operations[0]["operation"] == "unknown_gate"


def test_supported_qiskit_conversion_is_blocked_when_dependency_is_absent():
    result = convert_to_qiskit_circuit([("h", 0)])

    if importlib.util.find_spec("qiskit") is None:
        assert result.status == "BLOCKED"
        assert result.circuit is None
        assert "not installed" in result.error.lower()


def test_counts_convert_to_project_bit_order_probabilities():
    probabilities = counts_to_probabilities({"01": 3, "10": 1}, shots=4)

    assert probabilities == {"10": 0.75, "01": 0.25}


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        counts_to_probabilities({"0": -1})
    with pytest.raises(ValueError):
        counts_to_probabilities({"0": 0})


def test_offline_execution_uses_deterministic_fixture_and_never_claims_ibm():
    first = run_offline_validation([("h", 0), ("measure", 0)], shots=128, seed=19)
    second = run_offline_validation([("h", 0), ("measure", 0)], shots=128, seed=19)
    first_data = first.to_dict()
    second_data = second.to_dict()

    assert first.overall_status == "OFFLINE_TESTED"
    assert first.execution_origin == "OFFLINE_FIXTURE"
    assert first.real_hardware_executed is False
    assert first.observed_counts == second.observed_counts
    assert first.diagnosis == second.diagnosis
    assert first_data["real_ibm_status"] == "NOT_EXECUTED"
    assert first_data["calibration_snapshot"]["capture_source"] == "OFFLINE_FIXTURE"
    assert first_data["metrics"]["accuracy"] is None


def test_real_access_detection_and_execution_block_without_dependencies_or_credentials():
    access = check_ibm_access_status()
    if not access.qiskit_available or not access.runtime_available or not access.credentials_detected:
        result = execute_on_ibm_backend([("x", 0)], backend_name="user-selected", token="", shots=16)
        assert result.overall_status == "BLOCKED"
        assert result.real_hardware_executed is False
        assert result.job_id is None
        assert result.errors


def test_report_separates_offline_and_real_hardware_statuses():
    result = run_offline_validation()

    report = build_ibm_validation_report(result)

    assert report["phase_statuses"]["IMPLEMENTED"] == "IMPLEMENTED"
    assert report["phase_statuses"]["OFFLINE_TESTED"] == "OFFLINE_TESTED"
    assert report["phase_statuses"]["REAL_IBM_EXECUTED"] == "NOT_EXECUTED"
    assert report["calibration_retrieval_status"] == "OFFLINE_FIXTURE"
