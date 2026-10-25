"""Bounded simulator-independent characterization of the existing TVD guard."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np

from src.quantum_simulator.statistics import compare_distributions

SHOTS = (256, 1024, 4096)
BASE = (0.50, 0.25, 0.15, 0.10)
DELTAS = (0.025, 0.05, 0.10, 0.20)
NULL_REPLICATES = 2000
ALT_REPLICATES = 500
MAX_PAIRS = 12000
MAX_SECONDS = 600.0
MAX_MEMORY_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
PROTOCOL_ID = "synthetic-two-sample-tvd-v1-20261015"


def planned_pair_count() -> int:
    return len(SHOTS) * NULL_REPLICATES + len(SHOTS) * len(DELTAS) * ALT_REPLICATES


def distribution_for_delta(delta: float) -> tuple[float, ...]:
    if delta not in DELTAS:
        raise ValueError("delta must be one of the four preregistered alternatives")
    return (BASE[0] - delta, BASE[1] + delta, BASE[2], BASE[3])


def tvd(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("distributions must have four outcomes")
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def stream_seed(shots: int, condition_id: str, replicate: int, role: str) -> int:
    if role not in {"reference", "submitted"}:
        raise ValueError("invalid stream role")
    material = f"{PROTOCOL_ID}|{shots}|{condition_id}|{replicate}|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def pair_outcome(shots: int, condition_id: str, replicate: int, reference: tuple[float, ...], submitted: tuple[float, ...]) -> bool:
    reference_counts = np.random.default_rng(stream_seed(shots, condition_id, replicate, "reference")).multinomial(shots, reference)
    submitted_counts = np.random.default_rng(stream_seed(shots, condition_id, replicate, "submitted")).multinomial(shots, submitted)
    result = compare_distributions(reference_counts, submitted_counts, threshold=0.05, shots=shots)
    return bool(result.anomaly_detected)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """Binomial CDF with log-PMF terms to avoid large-combination overflow."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(probability) + (n - i) * math.log1p(-probability)
        for i in range(k + 1)
    ]
    peak = max(logs)
    return min(1.0, math.exp(peak) * sum(math.exp(value - peak) for value in logs))


def _cp_upper_bound(successes: int, trials: int, alpha: float) -> float:
    if successes == trials:
        return 1.0
    low, high = successes / trials, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _cp_lower_bound(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    low, high = 0.0, successes / trials
    for _ in range(64):
        middle = (low + high) / 2
        survival = 1.0 - _binomial_cdf(successes - 1, trials, middle)
        if survival < alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def interval(successes: int, trials: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact Clopper-Pearson interval."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= positive trials")
    lower = _cp_lower_bound(successes, trials, alpha / 2)
    upper = _cp_upper_bound(successes, trials, alpha / 2)
    return lower, upper


def one_sided_upper(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= positive trials")
    return _cp_upper_bound(successes, trials, alpha)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(prereg_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"create-only output already exists: {output_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    stats_path = script_path.parents[1] / "src" / "quantum_simulator" / "statistics.py"
    if prereg["artifacts"]["implementation_sha256"] != sha256(script_path):
        raise ValueError("implementation hash differs from frozen preregistration")
    if prereg["artifacts"]["statistics_module_sha256"] != sha256(stats_path):
        raise ValueError("statistics module hash differs from frozen preregistration")
    total = planned_pair_count()
    if total != prereg["resource_cap"]["planned_count_pairs"] or total > MAX_PAIRS:
        raise ValueError("planned pair count violates preregistration or hard cap")
    start = time.monotonic()
    cpu_start = time.process_time()
    tracemalloc.start()
    cells: dict[str, dict[str, Any]] = {}
    conditions = [("null", "null_equal", BASE, NULL_REPLICATES)]
    conditions += [("alternative", f"tvd_{int(delta * 1000):03d}", distribution_for_delta(delta), ALT_REPLICATES) for delta in DELTAS]
    processed = 0
    for shots in SHOTS:
        for kind, condition_id, submitted, reps in conditions:
            hits = 0
            for replicate in range(reps):
                hits += pair_outcome(shots, condition_id, replicate, BASE, submitted)
                processed += 1
                if processed > MAX_PAIRS:
                    raise RuntimeError("pair cap exceeded")
                if time.process_time() - cpu_start > MAX_SECONDS:
                    raise TimeoutError("10-minute CPU-time cap exceeded")
            rate = hits / reps
            row: dict[str, Any] = {"condition_id": condition_id, "kind": kind, "shots": shots, "replicates": reps, "detections": hits, "rate": rate, "tvd_truth": 0.0 if kind == "null" else float(condition_id[4:]) / 1000.0}
            if kind == "null":
                row["one_sided_95_upper"] = one_sided_upper(hits, reps)
            else:
                lo, hi = interval(hits, reps)
                row["two_sided_95_interval"] = [lo, hi]
            cells[f"{condition_id}|{shots}"] = row
            _, peak = tracemalloc.get_traced_memory()
            if peak > MAX_MEMORY_BYTES:
                raise MemoryError("256 MiB memory cap exceeded")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    null_rows = [cells[f"null_equal|{shots}"] for shots in SHOTS]
    passed = all(row["one_sided_95_upper"] <= 0.06 for row in null_rows)
    result = {
        "schema_version": "synthetic-two-sample-tvd-result-v1",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": sha256(prereg_path),
        "implementation_sha256": sha256(script_path),
        "statistics_module_sha256": sha256(stats_path),
        "resource_accounting": {"counting_convention": "one pair is one independent reference multinomial count vector plus one submitted multinomial count vector for a single condition and shot count", "planned_pairs": total, "actual_pairs": processed, "maximum_pairs": MAX_PAIRS, "workers": 1, "runtime_seconds": time.monotonic() - start, "cpu_seconds": time.process_time() - cpu_start, "tracemalloc_peak_bytes": peak},
        "cells": list(cells.values()),
        "null_gate": {"criterion": "one-sided exact 95% upper FPR bound <= 0.06 at each shot count", "passed": passed, "shot_results": [{"shots": row["shots"], "false_positives": row["detections"], "trials": row["replicates"], "fpr": row["rate"], "upper_95": row["one_sided_95_upper"]} for row in null_rows]},
        "verdict": "PASS" if passed else "FAIL",
        "limitations": ["Synthetic sampled distributions only; no circuit, simulator, QMutBench, family, program-fault, or hardware inference.", "Alternative power estimates are descriptive; no minimum power criterion was specified.", "Replicate intervals assume independently seeded multinomial count pairs."]
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise RuntimeError("256 MiB output cap exceeded")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(encoded)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prereg, args.output)
    print(json.dumps({"verdict": result["verdict"], "actual_pairs": result["resource_accounting"]["actual_pairs"], "output": str(args.output)}, sort_keys=True))

if __name__ == "__main__":
    main()
