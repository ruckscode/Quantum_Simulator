"""Smoke tests for the clean offline acceptance runner."""

from clean_simulation_test import build_report


def test_clean_acceptance_report_is_offline_and_reproducible():
    report = build_report()

    assert report["execution_mode"] == "LOCAL_OFFLINE_ONLY"
    assert report["circuit"]["shots_valid"] is True
    assert sum(report["circuit"]["local_simulator_counts"].values()) == report["shots"]
    assert report["ibm_integration"]["real_hardware_executed"] is False
    assert report["ibm_integration"]["real_ibm_status"] == "NOT_EXECUTED"
    assert report["diagnosis"]["program_fault_scenario"]["category"] == "PROGRAM_FAULT"
    assert report["diagnosis"]["hardware_anomaly_scenario"]["category"] == "HARDWARE_ANOMALY"
    assert report["diagnosis"]["hardware_anomaly_scenario"]["scenario_source"].startswith("synthetic")
