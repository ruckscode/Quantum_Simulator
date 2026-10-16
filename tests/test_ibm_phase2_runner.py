"""Safety tests for the opt-in IBM Phase 2 runner."""

import json

import ibm_phase2_runner as runner


def test_default_mode_never_discovers_backends_or_submits_jobs(monkeypatch, tmp_path):
    report_path = tmp_path / "ibm_phase2_report.json"
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_API_KEY", raising=False)
    monkeypatch.setattr(
        runner,
        "discover_ibm_backends",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not contact IBM by default")),
    )
    monkeypatch.setattr(
        runner,
        "execute_on_ibm_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not submit by default")),
    )

    assert runner.main(["--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["phase"] == "PHASE_2_IBM_HARDWARE_INTEGRATION"
    assert report["execution_mode"] == "NO_HARDWARE_JOB"
    assert report["real_hardware_executed"] is False


def test_backend_listing_without_explicit_token_is_blocked(monkeypatch, tmp_path):
    report_path = tmp_path / "ibm_phase2_report.json"
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_API_KEY", raising=False)
    monkeypatch.setattr(
        runner,
        "discover_ibm_backends",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not contact IBM without token")),
    )

    assert runner.main(["--list-backends", "--report", str(report_path)]) == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["operation_status"] == "BLOCKED"
    assert report["real_hardware_executed"] is False
