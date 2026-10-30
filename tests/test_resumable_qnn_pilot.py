"""Source-contract checks for registered resumable pilot tasks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_resumable_qnn_pilot as pilot  # noqa: E402


def test_registered_pilots_load_disjoint_validated_issue_group_tasks():
    task_ids = []
    record_sets = []
    for task in ("bugs4q_na_source_state", "bugs4q_core_source_state", "bugs4q_robust_execution",
                 "lintq_warning_triage_context_v2", "lintq_warning_triage_context_v2_balanced",
                 "quantum_platform_comment_v2_balanced", "qbugs_qasm_v2_source_state",
                 "mute_alg_hornbro_qasm_v2", "qolumbina_ideal_seeded_mutant_detection"):
        pilot.configure_task(task)
        rows, assignment, audit = pilot.read_task()
        spec = pilot.TASK_CONFIGS[task]
        assert len(rows) == spec["rows"]
        assert audit["record_count"] == spec["rows"]
        assert audit["group_count"] == spec["groups"]
        assert len(assignment) == spec["rows"]
        assert {int(row["label"]) for row in rows} == {0, 1}
        assert len({assignment[row["record_id"]][0] for row in rows}) == 5
        task_ids.append(task)
        record_sets.append({row["record_id"] for row in rows})
    assert len(task_ids) == len(set(task_ids))
    assert all(record_sets[i].isdisjoint(record_sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert record_sets[3] == record_sets[4]
    assert all(record_sets[6].isdisjoint(record_sets[i]) for i in range(6))
    assert all(record_sets[7].isdisjoint(record_sets[i]) for i in range(7))
    assert all(record_sets[8].isdisjoint(record_sets[i]) for i in range(8))
    assert len({pilot.TASK_CONFIGS[task]["output"] for task in task_ids}) == len(task_ids)


def test_registered_bugs4q_joint_and_source_restricted_transfer_tasks():
    outputs = set()
    expected_sources = {
        "bugs4q_joint_source_state": None,
        "bugs4q_core_only_transfer_source_restricted_v2": "Bugs4Q_core",
        "bugs4q_na_only_transfer_source_restricted_v2": "Bugs4Q_NA_additional",
    }
    for task, training_source in expected_sources.items():
        pilot.configure_task(task)
        rows, assignment, audit = pilot.read_task()
        assert len(rows) == 124
        assert audit["group_count"] == 62
        assert len(assignment) == 124
        assert {int(row["label"]) for row in rows} == {0, 1}
        assert len({fold for fold, _ in assignment.values()}) == 5
        spec = pilot.TASK_CONFIGS[task]
        assert spec.get("training_source") == training_source
        if training_source:
            assert {row["source"] for row in rows} == {"Bugs4Q_core", "Bugs4Q_NA_additional"}
        outputs.add(pilot.TASK_CONFIGS[task]["output"])
    assert len(outputs) == 3


def test_qolumbina_multibackend_pilots_use_validated_program_grouped_source():
    task_ids = (
        "qolumbina_multibackend_seeded_mutant_detection",
        "qolumbina_multibackend_seeded_mutant_detection_balanced",
    )
    outputs = set()
    for task in task_ids:
        pilot.configure_task(task)
        rows, assignment, audit = pilot.read_task()
        spec = pilot.TASK_CONFIGS[task]
        assert len(rows) == spec["rows"] == 546
        assert audit["group_count"] == spec["groups"] == 26
        assert len(assignment) == 546
        assert {int(row["label"]) for row in rows} == {0, 1}
        assert len({fold for fold, _ in assignment.values()}) == 5
        assert spec["feature_fields"] == ["code", "backend"]
        if task.endswith("_balanced"):
            assert spec.get("class_weighting") == "balanced"
        else:
            assert spec.get("class_weighting") is None
        outputs.add(spec["output"])
    assert len(outputs) == len(task_ids)


def test_atomic_checkpoint_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    import os
    import train_resource_light_qnn_tasks as light_qnn

    source = tmp_path / "checkpoint.tmp"
    destination = tmp_path / "checkpoint.npz"
    source.write_text("durable state", encoding="utf-8")
    original_replace = os.replace
    calls = {"count": 0}

    def transient_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("simulated transient Windows sharing lock")
        return original_replace(src, dst)

    monkeypatch.setattr(light_qnn.os, "replace", transient_replace, raising=False)
    monkeypatch.setattr(light_qnn.time, "sleep", lambda _: None)
    light_qnn.replace_with_retry(source, destination)
    assert calls["count"] == 3
    assert destination.read_text(encoding="utf-8") == "durable state"


def test_unregistered_pilot_task_is_rejected():
    try:
        pilot.configure_task("qmutbench_binary_mutation_outcome")
    except ValueError as exc:
        assert "Unregistered pilot task" in str(exc)
    else:
        raise AssertionError("unregistered task was accepted")


def test_balanced_weights_are_derived_from_training_labels_only():
    import numpy as np

    weights = pilot.training_fold_balanced_sample_weights(np.array([0, 0, 0, 1]))
    assert np.allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2.0])
    assert np.isclose(weights.mean(), 1.0)
    try:
        pilot.training_fold_balanced_sample_weights(np.array([1, 1, 1]))
    except ValueError as exc:
        assert "require both classes" in str(exc)
    else:
        raise AssertionError("single-class training fold was accepted")


def test_source_restricted_transfer_training_excludes_target_source_and_held_out_groups():
    rows = [
        {"record_id": f"{source}-{group}-{label}", "bug_id": f"{source}-{group}",
         "source": source, "label": label}
        for source in ("core", "na") for group in range(5) for label in (0, 1)
    ]
    assignment = {row["record_id"]: (int(row["bug_id"].rsplit("-", 1)[1]) + 1, row["bug_id"])
                  for row in rows}
    train, valid = pilot.build_fold_indices(rows, assignment, 2, "core")
    assert len(train) == 8
    assert len(valid) == 4
    assert {rows[i]["source"] for i in train} == {"core"}
    assert {rows[i]["source"] for i in valid} == {"core", "na"}
    assert {rows[i]["bug_id"] for i in train}.isdisjoint({rows[i]["bug_id"] for i in valid})
