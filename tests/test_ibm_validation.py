from types import SimpleNamespace
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
import inspect

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
    hardware_model_from_ibm_calibration,
    inspect_ibm_backend_calibration,
    preflight_ibm_backend,
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
    assert result["access"]["status"] == "SERVICE_PROVIDED"
    assert result["access"]["credentials_detected"] is False


def test_ibm_backend_selection_is_explicit_and_injected_service_needs_no_token():
    import ibm_validation

    assert inspect.signature(execute_on_ibm_backend).parameters["backend_name"].default is inspect.Parameter.empty
    assert inspect.signature(inspect_ibm_backend_calibration).parameters["backend_name"].default is inspect.Parameter.empty

    result = discover_ibm_backends(service=BackendServiceFixture())

    assert result["status"] == "PARSED_INJECTED_SERVICE"
    assert result["backends"][0]["name"] == "offline-fixture-backend"
    assert result["access"]["status"] == "SERVICE_PROVIDED"
    assert result["access"]["credentials_detected"] is False


def test_offline_validation_does_not_require_credentials_or_runtime(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(
        ibm_validation,
        "check_ibm_access_status",
        lambda: ibm_validation.IBMAccessStatus("BLOCKED", False, False, False, None, "no IBM account"),
    )
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: pytest.fail("offline path must not construct a Runtime service"))
    monkeypatch.setattr(ibm_validation, "_environment_runtime_service", lambda *args, **kwargs: pytest.fail("offline path must not use configured IBM accounts"))

    result = run_offline_validation([("h", 0), ("measure", 0)], shots=32, seed=7)

    assert result.overall_status == "OFFLINE_TESTED"
    assert result.real_hardware_executed is False


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


def test_backend_calibration_inspection_uses_snapshot_timestamp_without_live_job():
    expected_timestamp = datetime(2026, 10, 6, 9, 22, 1, tzinfo=timezone.utc)

    class TimestampBackend(BackendFixture):
        def properties(self):
            properties = super().properties()
            properties.last_update_date = expected_timestamp
            return properties

    class ServiceFixture:
        def backend(self, name):
            assert name == "offline-fixture-backend"
            return TimestampBackend()

    capture = inspect_ibm_backend_calibration("offline-fixture-backend", service=ServiceFixture())

    assert capture.backend_name == "offline-fixture-backend"
    assert capture.timestamp == expected_timestamp.isoformat()
    assert capture.n_qubits == 2
    assert capture.qubits[0]["readout_error"] == 0.02
    assert capture.qubits[0]["t1"] == 0.00012
    assert capture.qubits[0]["t2"] == 0.00008
    assert capture.gates[0]["error"] == 0.01
    assert capture.gates[0]["duration"] == 2e-7
    assert capture.coupling_map == [(0, 1)]


def test_ibm_calibration_maps_to_hardware_model_without_filling_missing_values():
    capture = capture_backend_calibration(BackendFixture())
    capture.timestamp = "2026-10-06T09:22:01+05:30"
    capture.coupling_map = [(0, 1)]
    capture.qubits[1].update(t1=None, t1_unit=None, t2=None, t2_unit=None)
    capture.gates.append({
        "gate": "reset",
        "qubits": [1],
        "error": None,
        "error_unit": None,
        "duration": 0.25,
        "duration_unit": "s",
        "supported": True,
    })

    model = hardware_model_from_ibm_calibration(capture)

    assert model.n_qubits == 2
    assert model.backend_name == "offline-fixture-backend"
    assert model.calibration_timestamp == capture.timestamp
    assert model.get_coupling_map() == [(0, 1)]
    assert model.get_directional_connectivity(0) == [1]
    assert model.get_directional_connectivity(1) == []
    assert model.get_qubit_properties(0).t1 == pytest.approx(120.0)
    assert model.get_qubit_properties(0).t2 == pytest.approx(80.0)
    assert model.get_qubit_properties(0).readout_error == 0.02
    assert model.get_qubit_properties(1).t1 is None
    assert model.get_qubit_properties(1).t2 is None
    assert model.get_qubit_properties(1).readout_error == 0.03
    cx = model.get_gate_calibration("cx", (0, 1))[0]
    assert cx.error == 0.01
    assert cx.duration == 2e-7
    reset = model.get_gate_calibration("reset", (1,))[0]
    assert reset.error is None
    assert reset.duration == 0.25
    snapshot = model.to_calibration_snapshot()
    assert snapshot.qubits[1]["t1"] is None
    assert snapshot.qubits[1]["t2"] is None
    assert snapshot.gates["reset"][0]["error"] is None


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
    with pytest.raises(ValueError, match="does not match requested shots"):
        counts_to_probabilities({"0": 3}, shots=4)
    with pytest.raises(ValueError, match="consistent measurement width"):
        counts_to_probabilities({"0": 2, "10": 2})
    with pytest.raises(ValueError, match="binary measurement bits"):
        counts_to_probabilities({"0x0": 4}, shots=4)
    with pytest.raises(ValueError, match="positive integer"):
        counts_to_probabilities({"0": 1}, shots=0)


def test_execution_interval_requires_valid_timezone_aware_ordered_timestamps():
    import ibm_validation

    ibm_validation._validate_execution_interval("2026-10-12T00:00:00+05:30", "2026-10-11T18:30:01Z")
    for start, end in ((None, "2026-10-12T00:00:00Z"), ("2026-10-12T00:00:00", "2026-10-12T00:00:01Z"), ("2026-10-12T00:00:02Z", "2026-10-12T00:00:01Z")):
        with pytest.raises(ValueError):
            ibm_validation._validate_execution_interval(start, end)


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
    assert first_data["calibration_snapshot"]["retrieved_at"]
    assert first_data["diagnosis_calibration_snapshot_id"] == first_data["calibration_snapshot_id"]
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


def _fresh_complete_backend_fixture(*, updated_at=None):
    backend = BackendFixture()
    properties = backend.properties()
    properties.last_update_date = updated_at or datetime.now(timezone.utc)
    backend.properties = lambda: properties
    return backend


def test_logical_to_physical_layout_maps_permutation_unused_qubits_and_multi_qubit_gate():
    import ibm_validation

    layout = SimpleNamespace(final_layout=None, final_index_layout=lambda **kwargs: [4, 2, 1])
    transpiled = SimpleNamespace(layout=layout, count_ops=lambda: {"cx": 1})
    info = ibm_validation._extract_transpilation_layout(transpiled, logical_num_qubits=3, backend_num_qubits=5)
    assert info["layout_status"] == "MAPPED"
    assert info["logical_to_physical"] == [4, 2, 1]

    logical = build_circuit_plan([("cx", 0, 2), ("measure", [0, 2], [0, 1])], num_qubits=3)
    physical = ibm_validation._map_plan_to_physical(logical, info["logical_to_physical"], backend_num_qubits=5)
    assert physical.num_qubits == 5
    assert physical.operations[0].qubits == (4, 1)
    assert physical.operations[1].qubits == (4, 1)

    model = HardwareModel(5, coupling_map=[(1, 4)])
    model.add_gate_calibration("cx", qubits=[4, 1], error=0.1)
    from quantum_simulator.hardware_fault_localizer import localize_hardware_fault
    localized = localize_hardware_fault(
        model,
        expected_circuit=[operation.to_tuple() for operation in physical.operations],
        actual_circuit=[operation.to_tuple() for operation in physical.operations],
    )
    assert any(candidate.component_type == "gate" and candidate.component == (4, 1) for candidate in localized.candidates)


@pytest.mark.parametrize("mapping", [[1, 1], [1], [1, 5]])
def test_invalid_logical_to_physical_layout_is_not_guessed(mapping):
    import ibm_validation

    transpiled = SimpleNamespace(layout=SimpleNamespace(final_layout=None, final_index_layout=lambda **kwargs: mapping), count_ops=lambda: {"cx": 1})
    result = ibm_validation._extract_transpilation_layout(transpiled, logical_num_qubits=2, backend_num_qubits=5)
    assert result["layout_status"] == "INVALID_LAYOUT"
    assert result["logical_to_physical"] is None


def test_routing_permutation_without_per_operation_map_is_insufficient_evidence():
    import ibm_validation

    missing = ibm_validation._extract_transpilation_layout(SimpleNamespace(count_ops=lambda: {"cx": 1}), logical_num_qubits=2, backend_num_qubits=2)
    assert missing["layout_status"] == "MISSING_LAYOUT"
    assert missing["logical_to_physical"] is None

    transpiled = SimpleNamespace(layout=SimpleNamespace(final_layout=object(), final_index_layout=lambda **kwargs: [0, 1]), count_ops=lambda: {"cx": 2})
    result = ibm_validation._extract_transpilation_layout(transpiled, logical_num_qubits=2, backend_num_qubits=2)
    assert result["layout_status"] == "ROUTING_LAYOUT_UNRESOLVED"
    assert result["logical_to_physical"] is None


def test_calibration_freshness_requires_timestamp_and_enforces_age():
    import ibm_validation

    capture = capture_backend_calibration(_fresh_complete_backend_fixture(updated_at=datetime(2026, 10, 12, tzinfo=timezone.utc)))
    now = datetime(2026, 10, 12, 0, 5, tzinfo=timezone.utc)
    assert ibm_validation.calibration_freshness(capture, max_age_seconds=600, now=now)["status"] == "FRESH"
    assert ibm_validation.calibration_freshness(capture, max_age_seconds=30, now=now)["status"] == "STALE"
    capture.timestamp = None
    assert ibm_validation.calibration_freshness(capture, now=now)["status"] == "MISSING_TIMESTAMP"


def test_calibration_freshness_normalizes_offsets_and_applies_900_second_boundary():
    import ibm_validation

    capture = capture_backend_calibration(_fresh_complete_backend_fixture())
    capture.timestamp = "2026-10-12T00:00:00+05:30"
    now = datetime(2026, 10, 11, 18, 45, tzinfo=timezone.utc)
    assert ibm_validation.calibration_freshness(capture, now=now)["status"] == "FRESH"
    now += timedelta(milliseconds=1)
    assert ibm_validation.calibration_freshness(capture, now=now)["status"] == "STALE"
    capture.timestamp = "2026-10-12T00:00:00"
    assert ibm_validation.calibration_freshness(capture, now=now)["status"] == "INVALID_TIMESTAMP"


def test_read_only_preflight_distinguishes_prerequisite_failures_and_fixture_success(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("BLOCKED", False, False, bool(token), "explicit token" if token else None, "runtime dependencies missing"))
    missing_deps = preflight_ibm_backend("mock-backend", token="fixture-token")
    assert missing_deps["status"] == "BLOCKED"
    assert missing_deps["blocker"] == "DEPENDENCIES_MISSING"
    assert missing_deps["checks"][0]["status"] == "FAIL"
    assert missing_deps["job_submitted"] is False

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("BLOCKED", True, True, False, None, "explicit credentials missing"))
    missing_credentials = preflight_ibm_backend("mock-backend")
    assert missing_credentials["status"] == "BLOCKED"
    assert missing_credentials["blocker"] == "CREDENTIALS_MISSING"

    class FailedService:
        def backend(self, name):
            raise RuntimeError("fixture backend lookup failed")

    backend_access = preflight_ibm_backend("offline-fixture-backend", service=FailedService())
    assert backend_access["status"] == "BLOCKED"
    assert backend_access["blocker"] == "BACKEND_ACCESS_FAILED"

    class ReadyService:
        def backend(self, name):
            return _fresh_complete_backend_fixture()

    now = datetime.now(timezone.utc) + timedelta(seconds=1)
    ready = preflight_ibm_backend("offline-fixture-backend", service=ReadyService(), max_calibration_age_seconds=30, now=now)
    assert ready["status"] == "PREFLIGHT_INJECTED_READY"
    assert ready["source"] == "CALLER_PROVIDED_SERVICE"
    assert ready["evidence_class"] == "INJECTED_OR_MOCKED"
    assert ready["real_backend_contact"] is None
    assert ready["job_submitted"] is False
    assert ready["transpilation_performed"] is False
    assert ready["calibration_assessment"]["completeness"] == "COMPLETE_FOR_REQUEST"
    assert all(check["status"] in {"PASS", "NOT_REQUIRED_FOR_INJECTED_SERVICE", "NOT_VERIFIED"} for check in ready["checks"])


def test_preflight_reports_stale_and_incomplete_calibration_without_claiming_readiness():
    class Service:
        def __init__(self, backend):
            self.selected = backend

        def backend(self, name):
            return self.selected

    stale_backend = _fresh_complete_backend_fixture(updated_at=datetime.now(timezone.utc) - timedelta(days=1))
    stale = preflight_ibm_backend("offline-fixture-backend", service=Service(stale_backend), max_calibration_age_seconds=10)
    assert stale["status"] == "BLOCKED"
    assert stale["blocker"] == "CALIBRATION_STALE"
    assert next(check for check in stale["checks"] if check["name"] == "calibration_timestamp")["status"] == "FAIL"
    assert stale["job_submitted"] is False

    incomplete_backend = _fresh_complete_backend_fixture()
    incomplete_properties = incomplete_backend.properties()
    incomplete_properties.qubits[0] = [entry for entry in incomplete_properties.qubits[0] if entry.name != "readout_error"]
    incomplete_backend.properties = lambda: incomplete_properties
    incomplete = preflight_ibm_backend("offline-fixture-backend", service=Service(incomplete_backend))
    assert incomplete["status"] == "BLOCKED"
    assert incomplete["blocker"] == "INCOMPLETE_CALIBRATION"
    assert incomplete["calibration"]["retrieval_status"] == "AVAILABLE"
    assert incomplete["calibration_assessment"]["completeness"] == "INCOMPLETE"


def test_read_only_preflight_blocks_unavailable_backend_and_malformed_topology():
    class Service:
        def __init__(self, backend):
            self.selected = backend

        def backend(self, name):
            return self.selected

    unavailable_backend = _fresh_complete_backend_fixture()
    unavailable_backend.status = lambda: SimpleNamespace(operational=False, status_msg="maintenance", pending_jobs=0)
    unavailable = preflight_ibm_backend("offline-fixture-backend", service=Service(unavailable_backend))
    assert unavailable["status"] == "BLOCKED"
    assert unavailable["blocker"] == "BACKEND_UNAVAILABLE"
    assert next(check for check in unavailable["checks"] if check["name"] == "backend_status")["status"] == "FAIL"

    malformed_backend = _fresh_complete_backend_fixture()
    malformed_backend.coupling_map = [[0, 4]]
    malformed = preflight_ibm_backend("offline-fixture-backend", service=Service(malformed_backend))
    assert malformed["status"] == "BLOCKED"
    assert malformed["blocker"] == "INCOMPLETE_CALIBRATION"
    assert malformed["calibration"]["errors"]
    assert next(check for check in malformed["checks"] if check["name"] == "qubit_and_coupling_indices")["status"] == "FAIL"


def test_mismatched_backend_identity_is_blocked_and_snapshot_ids_stay_associated():
    backend = _fresh_complete_backend_fixture()

    class Service:
        def backend(self, name):
            return backend

    result = preflight_ibm_backend("a-different-backend", service=Service())
    assert result["status"] == "BLOCKED"
    assert result["blocker"] == "BACKEND_IDENTITY_MISMATCH"
    assert result["backend"] == "offline-fixture-backend"
    assert result["requested_backend"] == "a-different-backend"
    assert result["calibration_snapshot_id"] == result["calibration_assessment"]["calibration_snapshot_id"]
    assert result["calibration_assessment"]["backend_name"] == result["calibration"]["backend_name"]


def test_fractional_coupling_qubit_index_is_not_truncated_to_valid_integer():
    backend = _fresh_complete_backend_fixture()
    backend.coupling_map = [(0, 1.5)]

    capture = capture_backend_calibration(backend)

    assert capture.coupling_map is None
    assert any("Malformed coupling map" in error for error in capture.errors)


def test_missing_multiqubit_calibration_is_reported_without_hardware_fault_attribution():
    from diagnosis_engine import diagnose_execution

    model = HardwareModel(2, coupling_map=[])
    result = diagnose_execution(
        expected_distribution={"00": 0.99, "11": 0.01},
        observed_distribution={"00": 0.01, "11": 0.99},
        expected_circuit=[("cx", 0, 1)],
        observed_circuit=[("cx", 0, 1)],
        hardware_model=model,
        shots=100,
    )

    assert result.hardware_evidence_candidates == []
    assert any("No gate calibration entry" in item for item in result.missing_evidence)
    assert any("No coupling-edge calibration" in item for item in result.missing_evidence)
    assert result.category.value == "AMBIGUOUS / INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    ("counts", "expected_error"),
    [({"0": 4}, "does not match requested shots"), ({"00": 5}, "measurement width is incompatible")],
)
def test_malformed_partial_execution_counts_fail_before_diagnosis(monkeypatch, counts, expected_error):
    import ibm_validation

    class Job:
        def job_id(self):
            return "mock-job"

        def result(self):
            return [SimpleNamespace(data=SimpleNamespace(c=SimpleNamespace(get_counts=lambda: counts)))]

    class Sampler:
        def __init__(self, *, mode):
            self.mode = mode

        def run(self, circuits, *, shots):
            return Job()

    runtime_package = SimpleNamespace(__path__=[])
    runtime_sampler = SimpleNamespace(Sampler=Sampler)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", runtime_package)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.executor_sampler", runtime_sampler)
    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))
    monkeypatch.setattr(ibm_validation, "ideal_distribution", lambda *args, **kwargs: {"0": 1.0})
    plan = build_circuit_plan([("x", 0), ("measure", 0)])
    monkeypatch.setattr(ibm_validation, "convert_to_qiskit_circuit", lambda circuit: ibm_validation.CircuitConversionResult("CONVERTED", object(), plan))
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: SimpleNamespace(backend=lambda name: _fresh_complete_backend_fixture()))
    monkeypatch.setattr(ibm_validation, "transpile_for_backend", lambda *args, **kwargs: {"status": "TRANSPILED", "circuit": object(), "layout_status": "MAPPED", "logical_to_physical": [0]})
    monkeypatch.setattr(ibm_validation, "diagnose_execution", lambda *args, **kwargs: pytest.fail("malformed result reached diagnosis"))

    result = execute_on_ibm_backend([("x", 0), ("measure", 0)], backend_name="offline-fixture-backend", token="fixture-token", shots=5)

    assert result.overall_status == "FAILED"
    assert result.diagnosis is None
    assert any(expected_error in error for error in result.errors)


def test_mismatched_backend_from_execution_service_blocks_before_transpilation(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))
    plan = build_circuit_plan([("x", 0)])
    monkeypatch.setattr(ibm_validation, "ideal_distribution", lambda *args, **kwargs: {"0": 1.0})
    monkeypatch.setattr(ibm_validation, "convert_to_qiskit_circuit", lambda circuit: ibm_validation.CircuitConversionResult("CONVERTED", object(), plan))
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: SimpleNamespace(backend=lambda name: _fresh_complete_backend_fixture()))
    monkeypatch.setattr(ibm_validation, "transpile_for_backend", lambda *args, **kwargs: pytest.fail("identity mismatch must block before transpilation"))

    result = execute_on_ibm_backend([("x", 0)], backend_name="requested-backend", token="fixture-token")

    assert result.overall_status == "BLOCKED"
    assert result.job_id is None
    assert result.calibration_assessment["backend_identity_status"] == "MISMATCH"
    serialized = result.to_dict()
    assert result.calibration_snapshot_id == serialized["calibration_snapshot_id"]
    assert result.calibration_assessment["calibration_snapshot_id"] == result.calibration_snapshot_id
    assert serialized["diagnosis_calibration_snapshot_id"] is None


def test_non_operational_unknown_status_and_invalid_shots_block_before_submission(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))
    plan = build_circuit_plan([("x", 0)])
    monkeypatch.setattr(ibm_validation, "ideal_distribution", lambda *args, **kwargs: {"0": 1.0})
    monkeypatch.setattr(ibm_validation, "convert_to_qiskit_circuit", lambda circuit: ibm_validation.CircuitConversionResult("CONVERTED", object(), plan))
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: pytest.fail("invalid shots must block before backend access"))

    invalid_shots = execute_on_ibm_backend([("x", 0)], backend_name="offline-fixture-backend", token="fixture-token", shots=True)
    assert invalid_shots.overall_status == "FAILED"
    assert invalid_shots.job_id is None

    backend = _fresh_complete_backend_fixture()
    backend.status = lambda: None
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: SimpleNamespace(backend=lambda name: backend))
    monkeypatch.setattr(ibm_validation, "transpile_for_backend", lambda *args, **kwargs: pytest.fail("unknown backend status must block before transpilation"))
    unknown_status = execute_on_ibm_backend([("x", 0)], backend_name="offline-fixture-backend", token="fixture-token", shots=8)
    assert unknown_status.overall_status == "BLOCKED"
    assert any("status is unknown" in error for error in unknown_status.errors)


def test_read_only_preflight_never_transpiles_or_executes(monkeypatch):
    import ibm_validation

    def forbidden(*args, **kwargs):
        pytest.fail("read-only preflight invoked an execution or transpilation API")

    monkeypatch.setattr(ibm_validation, "transpile_for_backend", forbidden)
    monkeypatch.setattr(ibm_validation, "execute_on_ibm_backend", forbidden)

    class Service:
        def backend(self, name):
            return _fresh_complete_backend_fixture()

    result = preflight_ibm_backend("offline-fixture-backend", service=Service())
    assert result["status"] == "PREFLIGHT_INJECTED_READY"
    assert result["job_submitted"] is False
    assert result["transpilation_performed"] is False


def test_execution_stops_before_submission_for_stale_calibration(monkeypatch):
    import ibm_validation

    monkeypatch.setattr(ibm_validation, "_explicit_credential_access_status", lambda token: ibm_validation.IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token"))
    monkeypatch.setattr(ibm_validation, "ideal_distribution", lambda *args, **kwargs: {"0": 1.0})
    plan = build_circuit_plan([("x", 0)])
    monkeypatch.setattr(ibm_validation, "convert_to_qiskit_circuit", lambda circuit: ibm_validation.CircuitConversionResult("CONVERTED", object(), plan))
    stale_backend = _fresh_complete_backend_fixture(updated_at=datetime.now(timezone.utc) - timedelta(days=1))
    monkeypatch.setattr(ibm_validation, "_runtime_service", lambda *args, **kwargs: SimpleNamespace(backend=lambda name: stale_backend))

    result = execute_on_ibm_backend([("x", 0)], backend_name="fixture", token="test-token", max_calibration_age_seconds=10)

    assert result.overall_status == "BLOCKED"
    assert result.job_id is None
    assert result.calibration_assessment["status"] == "STALE"
    assert any("no circuit was submitted" in error for error in result.errors)
