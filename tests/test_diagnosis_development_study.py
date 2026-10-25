"""Preflight and budget contracts for a future, not-yet-run diagnosis pilot."""

import hashlib
import json

import pytest

from experiments.diagnosis_development_study import (
    ResourceBudget,
    ResourceLimitExceeded,
    StudyBlockedError,
    StudyResourceLimits,
    audit_family_partitions,
    audit_materialized_sampling_frame,
    build_condition_manifest,
)
from experiments import diagnosis_lineage_inventory
from experiments.diagnosis_lineage_inventory import build_inventory, build_sampling_frame, discover_scan_roots, verify_hash


def _family(name, partition, topology):
    return {
        "family_id": name,
        "lineage_root_id": f"root-{name}",
        "topology_sha256": hashlib.sha256(topology.encode()).hexdigest(),
        "ancestor_root_ids": [],
        "partition": partition,
    }


def _inventory(*families, complete=True):
    return {
        "complete": complete,
        "families": [
            {"lineage_root_id": family["lineage_root_id"], "topology_sha256": family["topology_sha256"]}
            for family in families
        ],
    }


def test_incomplete_prior_inventory_fails_closed():
    family = _family("new-dev", "development", "new topology")
    with pytest.raises(StudyBlockedError, match="not certified complete"):
        audit_family_partitions([family], _inventory(_family("old", "development", "old"), complete=False))


def test_family_descendant_and_prior_topology_are_excluded():
    prior = _family("old", "development", "old topology")
    descendant = _family("new", "validation", "new topology")
    descendant["ancestor_root_ids"] = [prior["lineage_root_id"]]
    with pytest.raises(StudyBlockedError, match="descends from"):
        audit_family_partitions(
            [_family("dev", "development", "dev topology"), descendant], _inventory(prior)
        )
    reused = _family("reused", "validation", "old topology")
    with pytest.raises(StudyBlockedError, match="duplicates a previously inspected topology"):
        audit_family_partitions(
            [_family("dev", "development", "dev topology"), reused], _inventory(prior)
        )


def test_lineage_cannot_cross_development_and_validation():
    development = _family("dev", "development", "dev topology")
    validation = _family("val", "validation", "val topology")
    validation["lineage_root_id"] = development["lineage_root_id"]
    with pytest.raises(StudyBlockedError, match="must be unique"):
        audit_family_partitions(
            [development, validation],
            _inventory(_family("old", "development", "old topology")),
        )


def test_descendant_root_cannot_be_counted_as_a_separate_validation_family():
    development = _family("dev", "development", "dev topology")
    validation = _family("val", "validation", "mutated dev topology")
    validation["ancestor_root_ids"] = [development["lineage_root_id"]]
    with pytest.raises(StudyBlockedError, match="descendant of another proposed family root"):
        audit_family_partitions(
            [development, validation],
            _inventory(_family("old", "development", "old topology")),
        )


def test_condition_manifest_is_deterministic_and_keeps_family_partition():
    families = [
        _family("b", "validation", "topology b"),
        _family("a", "development", "topology a"),
    ]
    design = dict(strata=["healthy", "subtle"], shot_counts=[64, 128], repetitions=2, master_seed=91)
    first = build_condition_manifest(families, **design)
    second = build_condition_manifest(list(reversed(families)), **design)
    assert first == second
    assert len(first) == 16
    assert {row["partition"] for row in first if row["family_id"] == "a"} == {"development"}
    assert [row["seed"] for row in first] == [row["seed"] for row in second]


def test_resource_overrun_stops_without_partially_reserving_work():
    budget = ResourceBudget(StudyResourceLimits(4, 1000, 30.0, 128_000_000, 4096))
    budget.reserve(circuit_executions=2, shot_executions=500, output_bytes=1024)
    with pytest.raises(ResourceLimitExceeded, match="resource budget"):
        budget.reserve(circuit_executions=3, shot_executions=100, output_bytes=10)
    assert (budget.circuit_executions, budget.shot_executions, budget.output_bytes) == (2, 500, 1024)
    with pytest.raises(ResourceLimitExceeded, match="runtime limit"):
        budget.check_observed(runtime_seconds=31, peak_memory_bytes=64_000_000)
    with pytest.raises(ResourceLimitExceeded, match="memory limit"):
        budget.check_observed(runtime_seconds=1, peak_memory_bytes=129_000_000)


def test_valid_partition_requires_audited_old_roots_and_matching_fingerprints():
    prior = _family("old", "development", "old topology")
    frame = [_family("dev", "development", "fresh dev"), _family("val", "validation", "fresh val")]
    assert audit_family_partitions(frame, _inventory(prior)) == {"development": 1, "validation": 1}


def test_lineage_inventory_hashes_are_stable_and_exact_duplicates_are_linked(tmp_path):
    (tmp_path / "evidence").mkdir()
    artifact = '{"family_id":"f1","circuit_lineage_id":"recipe-a","circuit":[{"gate":"H","qubits":[0]}]}\n'
    (tmp_path / "evidence" / "a.json").write_text(artifact, encoding="utf-8")
    (tmp_path / "evidence" / "b.json").write_text(artifact, encoding="utf-8")
    first = build_inventory(tmp_path, ("evidence",))
    second = build_inventory(tmp_path, ("evidence",))
    assert first["inventory_sha256"] == second["inventory_sha256"]
    assert verify_hash(first, "inventory_sha256")
    assert first["summary"]["duplicate_content_count"] == 1
    assert first["summary"]["circuit_record_count"] == 2
    assert first["complete"] is False


def test_missing_roots_and_fingerprints_remain_unresolved_and_never_fill_frame(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "record.json").write_text(
        '{"family_id":"looks-new","circuit":[{"gate":"X","qubits":[0]}]}', encoding="utf-8"
    )
    inventory = build_inventory(tmp_path, ("evidence", "absent"))
    frame = build_sampling_frame(inventory)
    assert inventory["summary"]["unknown_lineage_circuit_count"] == 1
    assert inventory["missing_scan_roots"] == ["absent"]
    assert frame["candidates"] == []
    assert frame["shortfall"] == {"development": 8, "validation": 8}
    assert verify_hash(frame, "frame_sha256")
    with pytest.raises(StudyBlockedError, match="not certified complete"):
        audit_materialized_sampling_frame(inventory, frame)


def test_materialized_preflight_rejects_tampered_frame_and_inventory_mismatch(tmp_path):
    (tmp_path / "evidence").mkdir()
    inventory = build_inventory(tmp_path, ("evidence",))
    frame = build_sampling_frame(inventory)
    frame["status"] = "ready"
    with pytest.raises(StudyBlockedError, match="frame hash"):
        audit_materialized_sampling_frame(inventory, frame)


def test_frame_hash_and_sampler_are_bound_to_source_inventory(tmp_path):
    (tmp_path / "evidence").mkdir()
    inventory = build_inventory(tmp_path, ("evidence",))
    frame = build_sampling_frame(inventory)
    frame["inventory_sha256"] = "0" * 64
    with pytest.raises(StudyBlockedError, match="frame hash"):
        audit_materialized_sampling_frame(inventory, frame)


def test_inconsistent_family_root_mappings_are_preserved_as_review_flags(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "one.json").write_text(
        '{"family_id":"f","lineage_root_id":"r1","circuit":[{"gate":"H"}]}', encoding="utf-8"
    )
    (tmp_path / "evidence" / "two.json").write_text(
        '{"family_id":"f","lineage_root_id":"r2","circuit":[{"gate":"X"}]}', encoding="utf-8"
    )
    inventory = build_inventory(tmp_path, ("evidence",))
    assert any(
        flag["kind"] == "family_id_maps_to_multiple_roots" and flag["identifier"] == "f"
        for flag in inventory["conflicts"]
    )
    assert inventory["complete"] is False


def test_complete_boolean_without_hash_bound_independent_reconciliation_is_rejected(tmp_path):
    (tmp_path / "evidence").mkdir()
    inventory = build_inventory(tmp_path, ("evidence",))
    inventory["complete"] = True
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    frame = build_sampling_frame(inventory)
    with pytest.raises(StudyBlockedError, match="zero-unresolved metrics and a hash-bound"):
        audit_materialized_sampling_frame(inventory, frame)


def test_multiple_nominal_family_ids_share_one_prior_recipe_root(tmp_path):
    (tmp_path / "evidence").mkdir()
    for idx, family in enumerate(("nominal-1", "nominal-2")):
        (tmp_path / "evidence" / f"{idx}.json").write_text(
            json.dumps({"family_id": family, "circuit_lineage_id": "recipe-root-a",
                        "circuit": [{"gate": "H", "qubits": [0]}]}), encoding="utf-8"
        )
    inventory = build_inventory(tmp_path, ("evidence",))
    assert len(inventory["families"]) == 1
    assert inventory["families"][0]["lineage_root_id"] == "recipe-root-a"
    assert inventory["families"][0]["family_ids"] == ["nominal-1", "nominal-2"]
    assert inventory["families"][0]["eligibility"] == "excluded_prior_artifact"


def test_metadata_only_root_is_indexed_without_fabricating_a_fingerprint(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "manifest.json").write_text(
        '{"family_id":"known-family","circuit_lineage_id":"known-root"}', encoding="utf-8"
    )
    inventory = build_inventory(tmp_path, ("evidence",))
    assert len(inventory["families"]) == 1
    assert inventory["families"][0]["lineage_root_id"] == "known-root"
    assert inventory["families"][0]["topology_sha256"] is None
    assert inventory["complete"] is False


def test_scan_root_discovery_includes_retained_outputs_but_omits_active_test_temps(tmp_path):
    for name in ("data", ".pwb-retained-output", ".pytest-tmp-retained", ".t-lineage-current", ".pwb-clean-env-copy"):
        (tmp_path / name).mkdir()
    roots = discover_scan_roots(tmp_path)
    assert "data" in roots
    assert ".pwb-retained-output" in roots
    assert ".pytest-tmp-retained" in roots
    assert ".t-lineage-current" not in roots
    assert ".pwb-clean-env-copy" not in roots


def test_inventory_resource_cap_marks_scan_incomplete(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "a.json").write_text('{"a":1}', encoding="utf-8")
    (evidence / "b.json").write_text('{"b":2}', encoding="utf-8")
    monkeypatch.setattr(diagnosis_lineage_inventory, "MAX_SCAN_FILES", 1)
    inventory = build_inventory(tmp_path, ("evidence",))
    assert inventory["scan_truncated"] is True
    assert inventory["summary"]["artifact_count"] == 1
    assert inventory["complete"] is False


def test_source_declared_recipe_domain_is_recorded_separately_from_observed_roots(tmp_path):
    source = tmp_path / "experiments"
    source.mkdir()
    (source / "full_dataset.py").write_text(
        'circuit_lineage_id=f"design_recipe_{index % 6 + 1:02d}"', encoding="utf-8"
    )
    inventory = build_inventory(tmp_path, ("experiments",))
    rule = inventory["source_lineage_rules"][0]
    assert rule["declared_possible_roots"] == [f"design_recipe_{i:02d}" for i in range(1, 7)]
    assert inventory["summary"]["observed_lineage_root_count"] == 0
    assert inventory["summary"]["source_declared_possible_root_count"] == 6
    assert all(row["eligibility"] == "excluded_prior_artifact" for row in inventory["families"])
