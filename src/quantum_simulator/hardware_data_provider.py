"""Synthetic-by-default and optional live IBM Quantum calibration provider.

The provider fetches calibration metadata only; it does not submit circuits to
physical hardware. When no IBM token is configured, deterministic synthetic
calibration is used automatically.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .hardware_model import HardwareModel


def _param(parameters: Any, name: str) -> float | None:
    for item in parameters or []:
        if getattr(item, "name", None) == name:
            value = getattr(item, "value", None)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return None


def _seconds(value: Any, unit: Any, *, fallback_unit: str) -> float | None:
    """Normalize a calibration time to seconds.

    IBM property payloads have appeared with explicit units and with legacy
    unitless numeric values (T1/T2 in microseconds, gate_length in nanoseconds).
    Prefer explicit metadata; only use the documented field convention when
    the unit is absent. HardwareModel and the simulator use seconds throughout.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric < 0:
        return None

    normalized = str(unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    factors = {
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
        "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
        "us": 1e-6, "usec": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
        "ns": 1e-9, "nsec": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
        "ps": 1e-12,
    }
    if normalized in factors:
        return numeric * factors[normalized]
    if normalized:
        # Do not silently reinterpret an unfamiliar explicit unit.
        return None
    return numeric * factors[fallback_unit]


def _param_seconds(parameters: Any, name: str, *, fallback_unit: str) -> float | None:
    for item in parameters or []:
        if getattr(item, "name", None) == name:
            return _seconds(
                getattr(item, "value", None),
                getattr(item, "unit", None),
                fallback_unit=fallback_unit,
            )
    return None


def _synthetic_model(n_qubits: int, seed: int = 42) -> HardwareModel:
    """Generate repeatable, explicitly synthetic calibration values."""
    rng = np.random.default_rng(seed)
    model = HardwareModel(
        n_qubits,
        backend_name="synthetic_backend",
        calibration_timestamp=None,
        calibration_timestamp_provenance="synthetic",
        default_gate_error=0.003,
        default_gate_duration=2.0e-7,
    )
    for q in range(n_qubits):
        model.set_qubit_properties(
            q,
            t1=float(rng.uniform(70e-6, 180e-6)),
            t2=float(rng.uniform(50e-6, 140e-6)),
            readout_error=float(rng.uniform(0.008, 0.045)),
            status="available",
            available=True,
            calibration_source="synthetic",
        )
    # Linear topology is a simple deterministic stand-in, not real backend data.
    for q in range(n_qubits - 1):
        model.add_edge(q, q + 1, error=float(rng.uniform(0.004, 0.025)),
                       duration=float(rng.uniform(2e-7, 8e-7)),
                       metadata_source="synthetic")
        model.add_gate_calibration(
            "cx", qubits=(q, q + 1), error=float(rng.uniform(0.004, 0.025)),
            duration=float(rng.uniform(2e-7, 8e-7)), source="synthetic")
        model.add_gate_calibration(
            "cx", qubits=(q + 1, q), error=float(rng.uniform(0.004, 0.025)),
            duration=float(rng.uniform(2e-7, 8e-7)), source="synthetic")
    for q in range(n_qubits):
        for gate in ("id", "x", "sx", "rz", "h", "y", "z", "s", "t"):
            model.add_gate_calibration(
                gate, qubits=(q,), error=float(rng.uniform(0.0001, 0.004)),
                duration=float(rng.uniform(2e-8, 1.5e-7)), source="synthetic")
    return model


def _ibm_model(n_qubits: int, backend_name: str, token: str, instance: str | None) -> HardwareModel:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError(
            "IBM token is configured but qiskit-ibm-runtime is not installed. "
            "Install it with: pip install qiskit-ibm-runtime"
        ) from exc

    kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform", "token": token}
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)
    backend = service.backend(backend_name)
    props = backend.properties()
    if props is None:
        raise RuntimeError(f"Backend {backend_name!r} did not provide calibration properties.")

    physical_count = int(getattr(backend, "num_qubits", len(getattr(props, "qubits", []))))
    if n_qubits > physical_count:
        raise ValueError(
            f"Circuit uses {n_qubits} logical qubits, but backend {backend_name!r} "
            f"has only {physical_count} qubits."
        )

    last_update = getattr(props, "last_update_date", None)
    timestamp = last_update.isoformat() if last_update is not None else None
    model = HardwareModel(
        n_qubits,
        backend_name=str(getattr(backend, "name", backend_name)),
        calibration_timestamp=timestamp,
        calibration_timestamp_provenance="reported" if timestamp else "unknown",
        default_gate_error=0.0,
        default_gate_duration=0.0,
    )
    # Explicit mapping for this first implementation: logical q maps to physical q.
    # Only the circuit-sized prefix is imported, preventing huge simulator states.
    for q in range(n_qubits):
        qubit_parameters = props.qubits[q] if q < len(props.qubits) else []
        values = {getattr(p, "name", ""): getattr(p, "value", None)
                  for p in qubit_parameters}
        t1_seconds = _param_seconds(qubit_parameters, "T1", fallback_unit="us")
        t2_seconds = _param_seconds(qubit_parameters, "T2", fallback_unit="us")
        model.set_qubit_properties(
            q,
            t1=t1_seconds,
            t2=t2_seconds,
            readout_error=values.get("readout_error"),
            status="available",
            available=True,
            physical_qubit=q,
            calibration_source="IBM Quantum properties",
            time_unit_normalization="T1/T2 normalized to seconds",
        )

    for gate in getattr(props, "gates", []):
        physical_qubits = tuple(int(q) for q in getattr(gate, "qubits", ()))
        if not physical_qubits or max(physical_qubits) >= n_qubits:
            continue
        name = str(getattr(gate, "gate", "")).lower()
        parameters = getattr(gate, "parameters", [])
        error = _param(parameters, "gate_error")
        duration = _param_seconds(parameters, "gate_length", fallback_unit="ns")
        if name:
            model.add_gate_calibration(
                name, qubits=physical_qubits,
                error=0.0 if error is None else error,
                duration=0.0 if duration is None else duration,
                source="IBM Quantum properties",
                physical_qubits=list(physical_qubits),\n                duration_unit_normalization="gate_length normalized to seconds",
            )

    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        for pair in coupling_map:
            if len(pair) == 2 and max(pair) < n_qubits:
                model.add_edge(int(pair[0]), int(pair[1]), source="IBM backend coupling map")
    return model


def load_hardware_data(
    n_qubits: int,
    *,
    source: str = "auto",
    backend_name: str | None = None,
    token: str | None = None,
    instance: str | None = None,
    seed: int = 42,
) -> tuple[HardwareModel, dict[str, Any]]:
    """Load real IBM calibration when configured, otherwise synthetic data.

    source='auto': IBM if a token exists, otherwise synthetic.
    source='synthetic': force synthetic values.
    source='ibm': require a token and live IBM calibration.
    A configured but invalid IBM token fails loudly rather than silently
    pretending real data was used.
    """
    normalized = source.lower().strip()
    if normalized not in {"auto", "synthetic", "ibm"}:
        raise ValueError("hardware source must be 'auto', 'synthetic', or 'ibm'")
    configured_token = token or os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN")
    if normalized == "synthetic" or (normalized == "auto" and not configured_token):
        model = _synthetic_model(n_qubits, seed=seed)
        return model, {
            "source": "synthetic",
            "is_real_calibration": False,
            "backend_name": model.backend_name,
            "calibration_timestamp": None,
            "note": "Deterministic synthetic calibration; not live hardware data.",
        }
    if not configured_token:
        raise ValueError(
            "IBM calibration requested but no token found. Set QISKIT_IBM_TOKEN, "
            "or use hardware_source='synthetic'."
        )
    name = backend_name or os.getenv("QISKIT_IBM_BACKEND", "ibm_kingston")
    model = _ibm_model(n_qubits, name, configured_token, instance or os.getenv("QISKIT_IBM_INSTANCE"))
    return model, {
        "source": "ibm_quantum",
        "is_real_calibration": True,
        "backend_name": model.backend_name,
        "calibration_timestamp": model.calibration_timestamp,
        "logical_to_physical_qubit_mapping": {str(q): q for q in range(n_qubits)},
        "note": "Live calibration metadata fetched from IBM Quantum; circuit execution remains local.",
    }
