import hashlib
import json

import pytest

from experiments.ml_data_pilot import (
    _canonical_digest,
    _jsonl_bytes,
    generate_pilot,
    require_fit_eligible,
    validate_pilot,
)


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _save_jsonl_and_digest(root, name, rows):
    content = _jsonl_bytes(rows)
    (root / name).write_bytes(content)
    manifest_path = root / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][name]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _refresh_config_digest(root, manifest):
    digest = _canonical_digest(manifest["design"]["configuration_records"])
    manifest["design"]["configuration_sha256"] = digest
    manifest["source_provenance"]["config_sha256"] = digest
    (root / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_pilot_generation_is_deterministic_provenanced_and_fit_blocked(tmp_path):
    first = generate_pilot(tmp_path / "first")
    second = generate_pilot(tmp_path / "second")

    assert first["provenance_valid"] and first["structurally_valid"]
    assert not first["fit_eligible"]
    assert first["counts"] == {"observations": 9, "labels": 9, "families": 3, "lineages": 3, "near_duplicate_family_pairs": 0}
    for name in ("observations.jsonl", "synthetic_labels.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    manifest_a = json.loads((tmp_path / "first" / "pilot_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((tmp_path / "second" / "pilot_manifest.json").read_text(encoding="utf-8"))
    for key in ("schema_version", "dataset_id", "design", "features", "source_provenance", "runtime", "artifacts", "counts"):
        assert manifest_a[key] == manifest_b[key]
    assert [item["deterministic_fingerprint"] for item in manifest_a["internal_run_artifacts"]["records"]] == [
        item["deterministic_fingerprint"] for item in manifest_b["internal_run_artifacts"]["records"]
    ]
    assert manifest_a["ground_truth_status"] == "SYNTHETIC_ONLY_NOT_INDEPENDENT"
    assert manifest_a["model_fitting_allowed"] is False
    with pytest.raises(ValueError, match="not eligible for model fitting"):
        require_fit_eligible(tmp_path / "first")


def test_pilot_refuses_to_overwrite_existing_evidence(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "do-not-touch.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_pilot(target)
    assert sentinel.read_text(encoding="utf-8") == "preserved"


def test_pilot_rejects_source_and_runtime_mismatch(tmp_path):
    generate_pilot(tmp_path / "pilot")
    source_result = validate_pilot(tmp_path / "pilot", current_sources={"different.py": "0" * 64})
    runtime_result = validate_pilot(tmp_path / "pilot", current_runtime={"python_version": "0.0"})
    assert any("source fingerprints" in item for item in source_result["blockers"])
    assert any("runtime identity" in item for item in runtime_result["blockers"])
    assert not source_result["fit_eligible"] and not runtime_result["fit_eligible"]


def test_pilot_rejects_stale_configuration_and_changed_internal_run(tmp_path):
    root = tmp_path / "stale-config"
    generate_pilot(root)
    manifest_path = root / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["design"]["configuration_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert any("configuration records" in item for item in validate_pilot(root)["blockers"])

    root = tmp_path / "tampered-report"
    generate_pilot(root)
    report = next((root / "runs").glob("run-*.json"))
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("internal run artifact digest mismatch" in item for item in validate_pilot(root)["blockers"])


def test_pilot_keeps_each_lineage_inside_one_predeclared_partition(tmp_path):
    generate_pilot(tmp_path / "pilot")
    rows = _load_jsonl(tmp_path / "pilot" / "observations.jsonl")
    partitions = {}
    for row in rows:
        partitions.setdefault(row["circuit_lineage_id"], set()).add(row["partition"])
    assert all(len(values) == 1 for values in partitions.values())
    assert {next(iter(value)) for value in partitions.values()} == {"development", "calibration", "untouched_test"}


def test_pilot_rejects_gate_mismatch_leakage_and_incompatible_schema(tmp_path):
    root = tmp_path / "pilot"
    generate_pilot(root)
    rows = _load_jsonl(root / "observations.jsonl")
    rows[0]["predictors"]["gate_mismatch_fraction"] = 1.0
    _save_jsonl_and_digest(root, "observations.jsonl", rows)
    result = validate_pilot(root)
    assert any("incompatible predictor schema" in item for item in result["blockers"])
    assert not result["fit_eligible"]

    root = tmp_path / "incompatible"
    generate_pilot(root)
    rows = _load_jsonl(root / "observations.jsonl")
    rows[0]["predictors"].pop("observed_entropy_bits")
    _save_jsonl_and_digest(root, "observations.jsonl", rows)
    assert any("incompatible predictor schema" in item for item in validate_pilot(root)["blockers"])


def test_pilot_flags_near_duplicate_families_split_across_partitions(tmp_path):
    root = tmp_path / "pilot"
    generate_pilot(root)
    manifest = json.loads((root / "pilot_manifest.json").read_text(encoding="utf-8"))
    configs = manifest["design"]["configuration_records"]
    family_a, family_b = configs[0]["circuit_family_id"], configs[3]["circuit_family_id"]
    circuit = configs[0]["circuit"]
    for config in configs:
        if config["circuit_family_id"] == family_b:
            config["circuit"] = circuit
    _refresh_config_digest(root, manifest)
    result = validate_pilot(root)
    assert any("near-duplicate families cross partitions" in item for item in result["blockers"])
    assert any({item["left"], item["right"]} == {family_a, family_b} for item in result["near_duplicate_pairs"])


def test_pilot_handles_ambiguous_and_missing_labels_without_forcing_truth(tmp_path):
    root = tmp_path / "pilot"
    generate_pilot(root)
    labels = _load_jsonl(root / "synthetic_labels.jsonl")
    labels[0]["synthetic_scenario_label"] = "AMBIGUOUS"
    labels[0]["attribution_status"] = "OBSERVATIONALLY_EQUIVALENT"
    _save_jsonl_and_digest(root, "synthetic_labels.jsonl", labels)
    ambiguous = validate_pilot(root)
    assert "independent ground truth or adjudication evidence is unavailable" in ambiguous["fit_blockers"]
    assert not any("unsupported or missing synthetic label" in item for item in ambiguous["blockers"])

    root = tmp_path / "missing"
    generate_pilot(root)
    labels = _load_jsonl(root / "synthetic_labels.jsonl")[:-1]
    _save_jsonl_and_digest(root, "synthetic_labels.jsonl", labels)
    missing = validate_pilot(root)
    assert any("do not align one-to-one" in item for item in missing["blockers"])
    assert not missing["fit_eligible"]


def test_pilot_requires_source_runtime_and_adjudication_before_fit(tmp_path):
    root = tmp_path / "pilot"
    generate_pilot(root)
    manifest_path = root / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_provenance"].pop("source_sha256")
    manifest["runtime"] = {}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_pilot(root)
    assert not result["provenance_valid"]
    assert not result["fit_eligible"]
    with pytest.raises(ValueError, match="source fingerprints"):
        require_fit_eligible(root)
