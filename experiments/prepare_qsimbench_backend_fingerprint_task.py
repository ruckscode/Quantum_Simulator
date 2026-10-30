"""Prepare a leakage-safe Aer-vs-Kyiv classifier from QSimBench outcomes."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_quantum_datasets/qsimbench/dataset/dataset"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint"
NAME = re.compile(r"(?P<algorithm>[^_]+)_(?P<size>\d+)_(?P<backend>aer_simulator|fake_[^_]+)_(?P<run>\d+)\.json$")


def histogram(run_id: str) -> tuple[Counter[str], int]:
    path = DATA / "artifacts" / run_id / "history_circuit.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    counts: Counter[str] = Counter()
    records = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records += 1
            data = item.get("data", {})
            for outcome, amount in data.items():
                counts[str(outcome)] += int(amount)
    return counts, records


def bucket(x: float, bins: int = 10) -> int:
    return max(0, min(bins - 1, int(x * bins)))


def features(counts: Counter[str]) -> tuple[str, dict]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Empty output histogram")
    width = max(map(len, counts))
    probabilities = [v / total for v in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    entropy_norm = entropy / max(1.0, width)
    max_prob = max(probabilities)
    support_fraction = len(counts) / max(2.0, 2.0 ** min(width, 30))
    weights = [(sum(ch == "1" for ch in key), v / total) for key, v in counts.items()]
    mean_weight = sum(w * p for w, p in weights) / max(1, width)
    variance_weight = sum(((w / max(1, width) - mean_weight) ** 2) * p for w, p in weights)
    all_one = sum(p for key, p in zip(counts, probabilities) if key and all(ch == "1" for ch in key))
    values = {"entropy": entropy_norm, "max_probability": max_prob,
              "support_fraction": support_fraction, "mean_hamming_weight": mean_weight,
              "hamming_variance": variance_weight, "all_ones_probability": all_one,
              "bits": width, "shots": total}
    # Quantized named bins preserve feature identity without exposing raw outputs,
    # filenames, backend metadata, or circuit identity.
    text = " ".join(f"{name}_bin{bucket(values[name])}" for name in (
        "entropy", "max_probability", "support_fraction", "mean_hamming_weight",
        "hamming_variance", "all_ones_probability"))
    text += f" width_bin{min(width // 4, 15)}"
    return text, values


def main() -> None:
    metas = []
    for path in DATA.glob("*.json"):
        match = NAME.fullmatch(path.name)
        if match and match.group("backend") in {"aer_simulator", "fake_kyiv"}:
            metas.append({**match.groupdict(), "path": path})
    rows = []
    group_ids = {key: index + 1 for index, key in enumerate(sorted({(x["algorithm"], x["size"]) for x in metas}))}
    for meta in metas:
        counts, trace_records = histogram(meta["run"])
        code, desc = features(counts)
        label = int(meta["backend"] == "fake_kyiv")
        group_key = (meta["algorithm"], meta["size"])
        rows.append({"record_id": f"{meta['algorithm']}-{meta['size']}-{meta['backend']}",
                     "bug_id": group_ids[group_key], "label": label, "code": code,
                     "source": meta["backend"], "algorithm": meta["algorithm"],
                     "size": meta["size"], "trace_records_aggregated": trace_records,
                     "histogram_features": json.dumps(desc, sort_keys=True)})
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    frame = pd.DataFrame(rows).sort_values("record_id")
    frame.to_csv(OUT / "normalized_manifest.csv", index=False)
    task = {"status": "task_manifest_prepared",
            "task": "distinguish ideal Aer simulator output from fake_kyiv backend output after execution",
            "target": {"aer_simulator": 0, "fake_kyiv": 1},
            "records": int(len(frame)), "circuit_groups": int(frame.bug_id.nunique()),
            "algorithm_size_groups": int(frame.groupby(["algorithm", "size"]).ngroups),
            "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
            "group_split_unit": "algorithm-size circuit setting; both backend variants remain in the same fold",
            "feature_policy": "quantized descriptors of history_circuit measurement counts only; backend metadata, file paths, timestamps, stats traces, mirror-circuit traces, and circuit identity excluded",
            "trace_records_aggregated": int(frame.trace_records_aggregated.sum()),
            "limitations": ["Backend identity is predicted from post-execution measurement output; this is backend fingerprinting, not bug prediction.",
                            "Only the predeclared pair aer_simulator and fake_kyiv is used.",
                            "All 150 algorithm-size settings are grouped to measure generalization to held-out settings."]}
    (OUT / "task_definition.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    print(json.dumps(task, indent=2))


if __name__ == "__main__":
    main()
