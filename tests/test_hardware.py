import pytest
from datetime import datetime, timezone
import json

from quantum_simulator.hardware_model import (
    CalibrationSnapshot,
    HardwareModel,
    assess_calibration_freshness,
)


def test_hardware_model_creation():
    model = HardwareModel(3, coupling_map=[(0, 1), (1, 2)])
    assert model.get_qubit_count() == 3
    assert model.get_coupling_map() == [(0, 1), (1, 2)]


def test_qubit_calibration_data():
    model = HardwareModel(2)
    model.set_qubit_properties(0, t1=100.0, t2=80.0, readout_error=0.02, status="available")
    props = model.get_qubit_properties(0)
    assert props.t1 == 100.0
    assert props.t2 == 80.0
    assert props.readout_error == 0.02
    assert props.status == "available"


def test_gate_calibration_data():
    model = HardwareModel(2)
    model.add_gate_calibration("cx", qubits=(0, 1), error=0.03, duration=0.2, supported=True, source="calibration")
    gates = model.get_gate_calibration("cx", qubits=(0, 1))
    assert len(gates) == 1
    assert gates[0].error == 0.03
    assert gates[0].duration == 0.2
    assert gates[0].metadata["source"] == "calibration"


def test_readout_error_and_availability():
    model = HardwareModel(1)
    model.set_qubit_properties(0, readout_error=0.05, available=False, status="maintenance")
    props = model.get_qubit_properties(0)
    assert props.readout_error == 0.05
    assert props.available is False
    assert props.status == "maintenance"
    assert model.is_qubit_available(0) is False


def test_connectivity_and_directional_map():
    model = HardwareModel(4)
    model.add_edge(0, 1, direction="bidirectional", error=0.01, duration=0.07)
    model.add_edge(1, 2, direction="forward", error=0.02, duration=0.08)
    edge = model.get_edge(0, 1)
    assert edge is not None
    assert edge.direction == "bidirectional"
    assert edge.error == 0.01
    assert 2 in model.get_directional_connectivity(1)
    assert 0 in model.get_directional_connectivity(1)


def test_invalid_inputs():
    model = HardwareModel(2)
    with pytest.raises(ValueError):
        model.get_qubit_properties(2)
    with pytest.raises(ValueError):
        model.set_qubit_properties(-1)
    with pytest.raises(ValueError):
        model.add_edge(0, 0)
    with pytest.raises(ValueError):
        model.set_qubit_properties(0, readout_error=1.5)
    with pytest.raises(ValueError):
        model.set_qubit_properties(0, t1=-1)


def test_calibration_snapshot_round_trip():
    model = HardwareModel(
        2, default_gate_error=0.12, default_gate_duration=0.07,
        calibration_timestamp="2025-01-01T00:00:00+00:00",
        calibration_timestamp_provenance="reported",
    )
    model.set_qubit_properties(0, t1=25.0, readout_error=0.04)
    model.add_gate_calibration("rx", qubits=(0,), error=0.01, duration=0.03)
    model.add_edge(0, 1, error=0.02, duration=0.05)

    snapshot = model.to_calibration_snapshot()
    new_model = HardwareModel(2)
    new_model.from_snapshot(snapshot)

    assert new_model.get_qubit_properties(0).t1 == 25.0
    assert new_model.get_qubit_properties(0).readout_error == 0.04
    assert new_model.get_gate_calibration("rx", qubits=(0,))[0].error == 0.01
    assert new_model.get_edge(0, 1).error == 0.02
    assert new_model.default_gate_error == 0.12
    assert new_model.default_gate_duration == 0.07
    assert new_model.calibration_timestamp == model.calibration_timestamp
    assert new_model.calibration_timestamp_provenance == "reported"


def test_snapshot_distinguishes_acquisition_and_creation_timestamps_in_serialization():
    model = HardwareModel(
        1,
        calibration_timestamp="2025-01-01T00:00:00+00:00",
        calibration_timestamp_provenance="reported",
    )
    snapshot = model.to_calibration_snapshot()
    serialized = json.loads(json.dumps(snapshot.to_dict()))
    restored = CalibrationSnapshot.from_dict(serialized)
    assert restored.timestamp == "2025-01-01T00:00:00+00:00"
    assert restored.timestamp_provenance == "reported"
    assert restored.created_at == snapshot.created_at
    assert restored.created_at != restored.timestamp


def test_calibration_freshness_requires_explicit_policy_and_timezone_aware_time():
    now = datetime(2025, 1, 1, 0, 10, tzinfo=timezone.utc)
    timestamp = "2025-01-01T00:00:00-00:00"
    assert assess_calibration_freshness(
        timestamp, provenance="reported", now=now,
    )["status"] == "AGE_KNOWN_FRESHNESS_UNASSESSED"
    fresh = assess_calibration_freshness(
        "2025-01-01T00:05:00Z", provenance="reported",
        max_age_seconds=600, now=now,
    )
    assert fresh["status"] == "FRESH"
    assert fresh["age_seconds"] == 300
    stale = assess_calibration_freshness(
        timestamp, provenance="reported", max_age_seconds=599, now=now,
    )
    assert stale["status"] == "STALE"
    assert assess_calibration_freshness(
        "2025-01-01T00:00:00", provenance="reported", max_age_seconds=600, now=now,
    )["status"] == "TIMEZONE_REQUIRED"


@pytest.mark.parametrize(
    ("timestamp", "provenance", "expected"),
    [
        (None, "reported", "MISSING_TIMESTAMP"),
        ("synthetic-config", "synthetic", "SYNTHETIC"),
        ("bad timestamp", "reported", "INVALID_TIMESTAMP"),
        ("2025-01-01T00:00:00Z", "unknown", "UNKNOWN_PROVENANCE"),
        ("2025-01-01T00:01:00Z", "reported", "FUTURE_TIMESTAMP"),
    ],
)
def test_calibration_freshness_provenance_and_invalid_time_cases(timestamp, provenance, expected):
    now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    result = assess_calibration_freshness(
        timestamp, provenance=provenance, max_age_seconds=600, now=now,
    )
    assert result["status"] == expected


def test_synthetic_config_remains_explicitly_synthetic_and_missing_time_is_unknown():
    synthetic = HardwareModel(1, calibration_timestamp="synthetic-config")
    assert synthetic.calibration_timestamp_provenance == "synthetic"
    assert assess_calibration_freshness(
        synthetic.calibration_timestamp,
        provenance=synthetic.calibration_timestamp_provenance,
        max_age_seconds=1,
    )["status"] == "SYNTHETIC"
    legacy = CalibrationSnapshot(timestamp=None)
    assert legacy.timestamp is None
    assert legacy.timestamp_provenance == "unknown"


def test_phase_a_regression_gate_module():
    import quantum_simulator.gates as gates

    assert gates.get_gate("h") is gates.H
    assert gates.get_gate_qubit_count("cx") == 2
    assert gates.is_parameterized_gate("rx") is True


def test_phase_b_regression_simulator_module():
    from quantum_simulator import QuantumSimulator

    sim = QuantumSimulator(2)
    sim.run_circuit([("h", 0), ("cnot", 0, 1)])
    expected = [1 / 2**0.5, 0, 0, 1 / 2**0.5]
    assert abs(sim.state[0] - expected[0]) < 1e-12
    assert abs(sim.state[3] - expected[3]) < 1e-12
