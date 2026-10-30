"""Check optional PennyLane simulator backend forward/gradient parity."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pennylane as qml
import psutil


def _evaluate(name: str) -> dict:
    started = time.perf_counter()
    process = psutil.Process()
    before = process.memory_info().rss
    device = qml.device(name, wires=3, shots=None)

    diff_method = "adjoint" if name == "lightning.gpu" else "backprop"

    @qml.qnode(device, interface="autograd", diff_method=diff_method)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(3), rotation="Y")
        for wire in range(3):
            qml.Rot(*weights[wire], wires=wire)
        qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[1, 2]); qml.CNOT(wires=[2, 0])
        return tuple(qml.expval(qml.PauliZ(w)) for w in range(3))

    from pennylane import numpy as pnp
    from autograd import grad
    x = pnp.array([.11, .47, .89], requires_grad=False)
    w = pnp.array(np.arange(9, dtype=float).reshape(3, 3) / 13, requires_grad=True)
    loss = lambda value: pnp.sum(pnp.stack(circuit(x, value)) ** 2)
    values = np.asarray(circuit(x, w), dtype=float)
    gradient = np.asarray(grad(loss)(w), dtype=float)
    return {"device": name, "diff_method": diff_method, "status": "gradient_passed", "values": values.tolist(),
        "gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_before_bytes": before, "rss_after_bytes": process.memory_info().rss,
        "peak_rss_bytes": process.memory_info().peak_wset if hasattr(process.memory_info(), "peak_wset") else None}


def main():
    names = ["default.qubit", "lightning.qubit", "lightning.gpu"]
    results, reference, reference_gradient = [], None, None
    for name in names:
        try:
            item = _evaluate(name)
            if reference is None:
                reference = np.asarray(item["values"])
                reference_gradient = np.asarray(item["gradient"])
                item["parity_max_abs"] = 0.0
                item["usable"] = item["gradient_finite"]
            else:
                delta = float(np.max(np.abs(np.asarray(item["values"]) - reference)))
                gradient_delta = float(np.max(np.abs(np.asarray(item["gradient"]) - reference_gradient)))
                item["parity_max_abs"] = delta
                item["gradient_parity_max_abs"] = gradient_delta
                item["usable"] = item["gradient_finite"] and delta <= 1e-7 and gradient_delta <= 1e-6
            results.append(item)
        except Exception as exc:  # plugin absent or unsupported gradient path
            results.append({"device": name, "status": "unavailable", "usable": False,
                            "reason": f"{type(exc).__name__}: {exc}"})
    selected = os.environ.get("QMB_DEVICE", "default.qubit")
    payload = {"schema": "qmutbench-qml-backend-smoke-v1", "devices": results,
        "selected_device": selected,
        "selected_device_usable": any(row["device"] == selected and row.get("usable") for row in results),
        "gpu_available": any(row["device"] == "lightning.gpu" and row.get("usable") for row in results),
        "meaning": "simulator backend compatibility only; not a quantum-advantage test"}
    print(json.dumps(payload, indent=2))
    if not payload["selected_device_usable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
