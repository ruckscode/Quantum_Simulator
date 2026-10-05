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
