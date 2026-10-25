import three_way_validation as three_way
from ibm_validation import build_circuit_plan


def test_experiment_definitions_cover_requested_circuits():
    assert list(three_way.EXPERIMENTS) == ["X", "H", "Bell", "GHZ", "RX(pi)", "RY(pi)"]
    assert three_way.EXPERIMENTS["Bell"].instructions == (("h", 0), ("cx", 0, 1))
    assert three_way.EXPERIMENTS["GHZ"].n_qubits == 3
    assert all(definition.instructions for definition in three_way.EXPERIMENTS.values())


def test_runner_accepts_operation_list_and_writes_json_report(tmp_path, monkeypatch):
    monkeypatch.setattr(three_way.importlib.util, "find_spec", lambda name: None)
    report = three_way.run_three_way_validation("custom", n_qubits=1, instructions=[("x", 0)])
    destination = three_way.write_json_report(tmp_path / "custom.json", report)
    import json
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["circuit"]["name"] == "custom"
    assert loaded["circuit"]["instructions"] == [["x", 0]]
    assert loaded["environments"]["our_simulator"]["status"] == "COMPLETED"


def test_three_way_measurements_parse_with_same_qubit_classical_bit_mapping():
    measurements = three_way._measurements(3)

    plan = build_circuit_plan(measurements)

    assert plan.supported
    assert [(operation.qubits, operation.classical_bits) for operation in plan.operations] == [
        ((0, 1, 2), (0, 1, 2)),
    ]


def test_bell_and_ghz_simulator_counts_include_full_register(monkeypatch):
    monkeypatch.setattr(three_way.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        three_way,
        "check_ibm_access_status",
        lambda: type("Access", (), {
            "status": "BLOCKED",
            "reason": "IBM dependencies or credentials unavailable.",
            "to_dict": lambda self: {"status": self.status, "reason": self.reason},
        })(),
    )

    expected_outcomes = {"Bell": {"00", "11"}, "GHZ": {"000", "111"}}
    for name, outcomes in expected_outcomes.items():
        report = three_way.run_three_way_validation(name, shots=128, seed=17)
        simulator = report["environments"]["our_simulator"]
        assert simulator["status"] == "COMPLETED"
        assert set(simulator["raw_counts"]) == outcomes
        assert all(len(outcome) == report["circuit"]["n_qubits"] for outcome in simulator["raw_counts"])


def test_our_simulator_executes_circuit_and_report_has_schema(monkeypatch):
    monkeypatch.setattr(three_way.importlib.util, "find_spec", lambda name: None)
    report = three_way.run_three_way_validation("X")

    assert report["schema_version"] == "1.0"
    assert set(report) == {"schema_version", "circuit", "environments", "pairwise_comparisons"}
    assert report["circuit"]["shots"] == 1000
    assert report["circuit"]["seed"] == 1729
    assert report["circuit"]["measurement_mapping"] == {"0": 0}
    our = report["environments"]["our_simulator"]
    assert our["status"] == "COMPLETED"
    assert our["raw_counts"] == {"1": 1000}
    assert our["normalized_counts"] == {"1": 1000}
    assert our["probability_distribution"] == {"1": 1.0}


def test_unavailable_environments_have_no_fabricated_results(monkeypatch):
    monkeypatch.setattr(three_way.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        three_way,
        "check_ibm_access_status",
        lambda: type("Access", (), {
            "status": "BLOCKED",
            "reason": "IBM dependencies or credentials unavailable.",
            "to_dict": lambda self: {"status": self.status, "reason": self.reason},
        })(),
    )
    report = three_way.run_three_way_validation("H")
    for name in ("qiskit_aer", "ibm_quantum"):
        result = report["environments"][name]
        assert result["status"] == "UNAVAILABLE"
        assert result["reason"]
        assert result["raw_counts"] is None
        assert result["normalized_counts"] is None
        assert result["probability_distribution"] is None
    assert report["pairwise_comparisons"]["qiskit_vs_our_simulator"] is None
    assert report["pairwise_comparisons"]["ibm_vs_our_simulator"] is None
    assert report["pairwise_comparisons"]["ibm_vs_qiskit"] is None


def test_aer_dependency_detection_reports_missing_aer(monkeypatch):
    monkeypatch.setattr(
        three_way.importlib.util,
        "find_spec",
        lambda name: object() if name == "qiskit" else None,
    )
    result = three_way._run_aer([("h", 0)], 1, 1000, 1729)
    assert result["status"] == "UNAVAILABLE"
    assert result["raw_counts"] is None
    assert result["reason"] == "Qiskit Aer is not installed."


def test_aer_diagnosis_bridge_routes_synthetic_evidence_and_abstains(monkeypatch):
    from quantum_simulator.diagnosis_engine import DiagnosisCategory

    def fake_run(instructions, n_qubits, shots, seed, *, noise_config=None):
        flips = (noise_config or {}).get("readout_error", 0.0) == 1.0
        bit = int(any(tuple(op)[0] == "x" for op in instructions)) ^ int(flips)
        counts = {"0": shots, "1": 0} if bit == 0 else {"0": 0, "1": shots}
        return {
            "status": "COMPLETED", "reason": None, "normalized_counts": counts,
            "probability_distribution": {key: value / shots for key, value in counts.items()},
        }

    monkeypatch.setattr(three_way, "_run_aer", fake_run)
    normal = three_way.run_aer_diagnosis([], 1, shots=64, seed=19)
    program = three_way.run_aer_diagnosis([], 1, executed_instructions=[("x", 0)], shots=64, seed=19)
    hardware = three_way.run_aer_diagnosis([], 1, noise_config={"readout_error": 1.0}, shots=64, seed=19)
    ambiguous = three_way.run_aer_diagnosis(
        [], 1, executed_instructions=[("x", 0)], noise_config={"readout_error": 1.0}, shots=64, seed=19,
    )

    assert normal["diagnosis"].category is DiagnosisCategory.NO_ANOMALY
    assert program["diagnosis"].category is DiagnosisCategory.PROGRAM_FAULT
    assert hardware["diagnosis"].category is DiagnosisCategory.HARDWARE_ANOMALY
    assert hardware["calibration_evidence"]["provenance"] == "synthetic_simulation"
    assert ambiguous["diagnosis"].category is DiagnosisCategory.AMBIGUOUS
    assert ambiguous["diagnosis"].anomaly_detected is False
    assert normal["shots"] == program["shots"] == hardware["shots"] == 64


def test_aer_diagnosis_rejects_unsupported_or_invalid_noise_without_aer():
    import pytest

    with pytest.raises(ValueError, match="unsupported Aer noise"):
        three_way.run_aer_diagnosis([], 1, noise_config={"depolarizing_error": 0.1})
    with pytest.raises(ValueError, match="readout_error"):
        three_way.run_aer_diagnosis([], 1, noise_config={"readout_error": 1.1})


def test_aer_diagnosis_runtime_cases_when_aer_is_installed():
    import pytest

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from quantum_simulator.diagnosis_engine import DiagnosisCategory

    normal = three_way.run_aer_diagnosis([], 1, shots=128, seed=1729)
    program = three_way.run_aer_diagnosis([], 1, executed_instructions=[("x", 0)], shots=128, seed=1729)
    hardware = three_way.run_aer_diagnosis([], 1, noise_config={"readout_error": 1.0}, shots=128, seed=1729)

    assert normal["status"] == program["status"] == hardware["status"] == "COMPLETED"
    assert normal["diagnosis"].category is DiagnosisCategory.NO_ANOMALY
    assert program["diagnosis"].category is DiagnosisCategory.PROGRAM_FAULT
    assert hardware["diagnosis"].category is DiagnosisCategory.HARDWARE_ANOMALY


def test_fixed_width_distribution_includes_unobserved_outcomes_as_zero_probability():
    assert three_way._complete_distribution({"11": 1.0}, n_qubits=2) == {
        "00": 0.0,
        "01": 0.0,
        "10": 0.0,
        "11": 1.0,
    }
