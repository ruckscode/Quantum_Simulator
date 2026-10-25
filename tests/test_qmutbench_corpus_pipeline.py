import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments import qmutbench_corpus_pipeline as pipeline


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_corpus(tmp_path: Path, mutant_qasm: str | None = None, rows: int = 1):
    commit = "test-pinned-commit"
    manifest_path = tmp_path / "source_manifest.json"
    source_csv = tmp_path / "official_metadata.csv"
    qasm_root = tmp_path / "raw" / f"qasm_{commit}"
    original_rel = "Origin_programs/ghz_1_qubits.qasm"
    mutant_rel = "Mutated_programs/Mutants_ghz_1_qubits/AddGate_id_inGap_1_.qasm"
    original = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    mutant = mutant_qasm or "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nh q[0];\nid q[0];\nmeasure q[0] -> c[0];\n"
    for rel, text in ((original_rel, original), (mutant_rel, mutant)):
        path = qasm_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    fields = pipeline.EXPECTED_COLUMNS
    metadata_rows = []
    for index in range(rows):
        metadata_rows.append({
            "": str(index), "Origin_program": "ghz_indep_qiskit_1", "algorithm": "ghz",
            "qubits": "1", "gates": "1", "operation": "Add", "Position_int": "1",
            "Killed": "WOO", "gate": "id", "algorithm_group": "test",
            "dominant_state": "0", "position": "91-100%",
        })
    with source_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata_rows)
    qasm_entries = []
    for rel in (original_rel, mutant_rel):
        path = qasm_root / rel
        qasm_entries.append({"path": rel, "sha256": _digest(path), "bytes": path.stat().st_size})
    manifest = {
        "dataset": "QMutBench test fixture only", "source_commit": commit,
        "metadata_csv": {
            "path": str(source_csv), "sha256": _digest(source_csv),
            "bytes": source_csv.stat().st_size, "url": "fixture://not-a-real-family",
            "schema": fields,
        },
        "qasm_artifacts": {"pinned_commit": commit, "files": qasm_entries},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, source_csv, qasm_root


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_official_source_hash_schema_and_tamper_gate(tmp_path):
    manifest, source, _ = _fixture_corpus(tmp_path)
    verified, resolved_source, _, source_hash = pipeline.load_and_verify_source(manifest)
    assert resolved_source == source
    assert source_hash == verified["metadata_csv"]["sha256"]
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline.load_and_verify_source(manifest)


def test_metadata_schema_must_match_the_source_manifest(tmp_path):
    manifest_path, source, _ = _fixture_corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source.write_text("wrong,columns\n1,2\n", encoding="utf-8")
    manifest["metadata_csv"]["sha256"] = _digest(source)
    manifest["metadata_csv"]["bytes"] = source.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        pipeline.load_and_verify_source(manifest_path)


def test_lineage_and_stable_record_identifiers_are_preserved(tmp_path):
    manifest, _, _ = _fixture_corpus(tmp_path)
    m, source, qasm_root, source_hash = pipeline.load_and_verify_source(manifest)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    row[""] = "9"
    one = pipeline._process_row(row, 9, m, source_hash, qasm_root, 64, 10, {})
    two = pipeline._process_row(row, 9, m, source_hash, qasm_root, 64, 10, {})
    assert one[0]["record_id"] == two[0]["record_id"]
    assert one[0]["record_id"].endswith("row:000009")
    assert one[0]["lineage_group_id"] == "ghz_indep_qiskit_1"
    assert one[0]["original_record_id"] == one[1]["record_id"]
    assert one[0]["family_root_independence_verified"] is False
    assert one[0]["diagnosis_category_compatible"] is False


@pytest.mark.parametrize(
    ("qasm", "expected_status"),
    [
        ("OPENQASM 2.0;\nqreg q[1];\nthis is malformed;\n", "PARSE_REJECTED"),
        ("OPENQASM 2.0;\nqreg q[1];\nmystery q[0];\n", "UNSUPPORTED_OPERATION"),
    ],
)
def test_malformed_and_unsupported_qasm_are_retained_as_failures(tmp_path, qasm, expected_status):
    manifest, _, _ = _fixture_corpus(tmp_path, mutant_qasm=qasm)
    result = pipeline.run_ingestion(
        source_manifest_path=manifest, output_dir=tmp_path / "run",
        start_row_index=0, limit=1, shots=32, seed_base=12,
    )
    failures = _read_jsonl(tmp_path / "run/rejections_and_failures.jsonl")
    normalized = _read_jsonl(tmp_path / "run/normalized_records.jsonl")
    assert result["counts"]["failures"] == 1
    assert failures[0]["status"] == expected_status
    assert normalized[-1]["execution_status"] == expected_status
    assert result["observation_records"] == 0


def test_equivalent_mutant_is_marked_as_observationally_equivalent(tmp_path):
    manifest, _, _ = _fixture_corpus(tmp_path)
    result = pipeline.run_ingestion(
        source_manifest_path=manifest, output_dir=tmp_path / "run",
        start_row_index=0, limit=1, shots=64, seed_base=13,
    )
    observation = _read_jsonl(tmp_path / "run/simulated_observations.jsonl")[0]
    assert result["counts"]["equivalent_at_computational_basis"] == 1
    assert observation["exact_tvd"] == pytest.approx(0.0)
    assert observation["exact_observable_status"] == "equivalent_at_computational_basis"
    assert observation["noise_settings"]["hardware_model"] is None


def test_bounded_resume_recovers_from_interrupted_second_record(tmp_path, monkeypatch):
    manifest, _, _ = _fixture_corpus(tmp_path, rows=2)
    original_process = pipeline._process_row

    def interrupt_second(row, source_index, *args, **kwargs):
        if source_index == 1:
            raise RuntimeError("simulated process interruption")
        return original_process(row, source_index, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_process_row", interrupt_second)
    output = tmp_path / "resumable"
    with pytest.raises(RuntimeError, match="interruption"):
        pipeline.run_ingestion(source_manifest_path=manifest, output_dir=output, limit=2, shots=32)
    monkeypatch.setattr(pipeline, "_process_row", original_process)
    result = pipeline.run_ingestion(
        source_manifest_path=manifest, output_dir=output, limit=2, shots=32, resume=True,
    )
    assert result["counts"]["processed_source_records"] == 2
    assert len(_read_jsonl(output / "simulated_observations.jsonl")) == 2
    assert len(_read_jsonl(output / "normalized_records.jsonl")) == 3  # one original plus two mutants


def test_preflight_blocks_missing_provenance_overlap_and_label_leakage():
    records = [
        {"record_id": "m1", "source_csv_sha256": "a", "source_record_sha256": "b",
         "qasm_sha256": "c", "artifact_role": "generated_mutant", "lineage_group_id": "root-a"},
    ]
    result = pipeline.evaluation_preflight(
        records,
        partitions={"train": ["root-a"], "calibration": ["root-a"], "test": []},
        primary_features=["total_variation_distance", "mutation_id"],
        task_target="benchmark_mutant_artifact_detection",
        minimum_independent_families=3, independent_roots_verified=False,
        labels_independently_verified=False,
    )
    assert result["ready"] is False
    assert any("overlap" in blocker for blocker in result["blockers"])
    assert any("leaking" in blocker for blocker in result["blockers"])
    assert any("below required minimum" in blocker for blocker in result["blockers"])
    assert any("not externally verified" in blocker for blocker in result["blockers"])


def test_missing_qasm_provenance_is_not_promoted_to_eligible():
    result = pipeline.evaluation_preflight(
        [{"record_id": "m", "source_csv_sha256": "", "source_record_sha256": "",
          "artifact_role": "generated_mutant", "lineage_group_id": "root"}],
        partitions={"train": ["root"], "calibration": [], "test": []},
        primary_features=["total_variation_distance"],
        task_target="program_fault_detection", minimum_independent_families=None,
        independent_roots_verified=False, labels_independently_verified=False,
    )
    assert result["ready"] is False
    assert any("source hashes/provenance missing" in blocker for blocker in result["blockers"])
    assert any("only compatible" in blocker for blocker in result["blockers"])

