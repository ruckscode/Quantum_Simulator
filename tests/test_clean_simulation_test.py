"""Smoke tests for the clean offline acceptance runner."""

from clean_simulation_test import build_report


def test_clean_acceptance_report_is_offline_and_reproducible():
    report = build_report()

    assert report["execution_mode"] == "LOCAL_OFFLINE_ONLY"
    assert report["circuit"]["shots_valid"] is True
    assert sum(report["circuit"]["local_simulator_counts"].values()) == report["shots"]
    assert report["ibm_integration"]["real_hardware_executed"] is False
    assert report["ibm_integration"]["real_ibm_status"] == "NOT_EXECUTED"
    assert report["ibm_integration"]["offline_adapter_status"] == "OFFLINE_TESTED"

    # Qiskit is an optional comparison dependency. A clean core installation
    # must pass without installing it, while an installed Qiskit path must
    # actually perform the comparison.
    qiskit = report["qiskit_comparison"]
    assert qiskit["status"] in {"COMPARED", "NOT_INSTALLED"}
    assert qiskit["comparison_performed"] is (qiskit["status"] == "COMPARED")
    if qiskit["status"] == "COMPARED":
        assert qiskit["counts"]
        assert qiskit["total_variation_distance"] is not None
    else:
        assert qiskit["counts"] is None
        assert qiskit["total_variation_distance"] is None

    assert report["diagnosis"]["program_fault_scenario"]["category"] == "PROGRAM_FAULT"
    assert report["diagnosis"]["hardware_anomaly_scenario"]["category"] == "HARDWARE_ANOMALY"
    assert report["diagnosis"]["hardware_anomaly_scenario"]["scenario_source"].startswith("synthetic")
