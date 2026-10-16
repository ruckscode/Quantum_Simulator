"""Explicit, opt-in IBM Quantum Phase 2 runner.

Safe by default: without --list-backends or --submit, this script only reports
local dependency/credential readiness and never contacts IBM or submits a job.

Credentials must be supplied via environment variables, preferably a notebook
secret: QISKIT_IBM_TOKEN, IBM_QUANTUM_TOKEN, or QISKIT_IBM_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ibm_validation import (
    check_ibm_access_status,
    discover_ibm_backends,
    execute_on_ibm_backend,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "reports" / "ibm_phase2_report.json"
TOKEN_ENV_NAMES = ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN", "QISKIT_IBM_API_KEY")
INSTANCE_ENV_NAMES = ("QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE")
BELL_CIRCUIT = [("h", 0), ("cx", 0, 1), ("measure", [0, 1])]


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _base_report() -> dict[str, Any]:
    access = check_ibm_access_status().to_dict()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": "PHASE_2_IBM_HARDWARE_INTEGRATION",
        "execution_mode": "NO_HARDWARE_JOB",
        "access_status": access,
        "real_hardware_executed": False,
        "note": "No live request or hardware job is made unless --list-backends or --submit is explicitly provided.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list-backends", action="store_true", help="Contact IBM to list accessible backends; does not submit a job.")
    action.add_argument("--submit", action="store_true", help="Submit one Bell-circuit job to the explicitly named backend.")
    parser.add_argument("--backend", help="Exact backend name; required with --submit.")
    parser.add_argument("--shots", type=int, default=512, help="Hardware shots (1–4096; default 512).")
    parser.add_argument("--channel", default="ibm_quantum_platform", help="Qiskit Runtime channel.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report output path.")
    args = parser.parse_args(argv)

    if args.submit and not args.backend:
        parser.error("--submit requires --backend with an exact backend name.")
    if not 1 <= args.shots <= 4096:
        parser.error("--shots must be between 1 and 4096.")
    if not args.submit and args.backend:
        parser.error("--backend is only accepted with --submit.")
    if not args.submit and args.shots != 512:
        parser.error("--shots is only relevant with --submit.")

    token = _first_env(TOKEN_ENV_NAMES)
    instance = _first_env(INSTANCE_ENV_NAMES)
    report = _base_report()

    if not args.list_backends and not args.submit:
        report["next_action"] = (
            "Install qiskit and qiskit-ibm-runtime in this environment and set a token through the notebook's secret/environment mechanism."
            if report["access_status"]["status"] == "BLOCKED"
            else "Run with --list-backends to verify account access, then select an exact backend and explicitly use --submit."
        )
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        print(f"Report: {args.report}")
        return 0

    if not token:
        report["operation_status"] = "BLOCKED"
        report["error"] = "No explicit IBM token found in QISKIT_IBM_TOKEN, IBM_QUANTUM_TOKEN, or QISKIT_IBM_API_KEY."
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        print("No hardware job was submitted.")
        return 2

    if args.list_backends:
        result = discover_ibm_backends(token=token, instance=instance, channel=args.channel)
        report["operation"] = "LIST_BACKENDS"
        report["operation_status"] = result.get("status")
        report["backends"] = result.get("backends", [])
        report["error"] = result.get("error")
        report["execution_mode"] = "IBM_BACKEND_DISCOVERY"
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        print(f"Report: {args.report}")
        return 0 if result.get("status") == "DISCOVERED" else 2

    result = execute_on_ibm_backend(
        BELL_CIRCUIT,
        backend_name=args.backend,
        token=token,
        instance=instance,
        shots=args.shots,
        channel=args.channel,
    )
    payload = result.to_dict()
    report.update(payload)
    report["operation"] = "SUBMIT_BELL_CIRCUIT"
    report["execution_mode"] = "EXPLICIT_IBM_SUBMISSION"
    report["requested_backend"] = args.backend
    report["real_hardware_executed"] = bool(result.real_hardware_executed)
    report["warning"] = "This action may incur queue time and consume available IBM Quantum usage."
    _write_report(args.report, report)
    print(json.dumps(report, indent=2))
    print(f"Report: {args.report}")
    return 0 if result.real_hardware_executed else 2


if __name__ == "__main__":
    raise SystemExit(main())
