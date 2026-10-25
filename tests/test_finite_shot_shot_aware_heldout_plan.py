"""Focused metadata-only checks for the shot-aware held-out preregistration."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import finite_shot_shot_aware_heldout_plan as plan


def test_workload_and_configuration_generation_are_deterministic():
    assert plan.workload() == {"configurations": 9000, "shot_circuit_executions": 32256000}
    first = plan.build_heldout_ledger()
    second = plan.build_heldout_ledger()
    assert first == second
    assert len({row["configuration_id"] for row in first}) == 9000
    assert len({row["seed"] for row in first}) == 9000
    assert len({row["template_id"] for row in first}) == 600


def test_split_identity_seed_and_fingerprint_disjointness():
    result = plan.check_disjointness()
    assert result == {
        "prior_templates_checked": 240, "new_templates": 600,
        "new_configurations": 9000, "prior_sampling_seeds_checked": 3600,
        "new_sampling_seeds": 9000,
    }


def test_candidate_rule_and_acceptance_criteria_are_frozen():
    criteria = plan.frozen_criteria()
    assert criteria["candidate_cutoffs"] == {
        256: 0.18359375, 1024: 0.09375, 4096: 0.0478515625,
    }
    assert criteria["sensitivity_minimums"] == {
        "subtle": {256: 0.50, 1024: 0.80, 4096: 0.90},
        "moderate": {256: 0.80, 1024: 0.90, 4096: 0.95},
        "large": {256: 0.95, 1024: 0.95, 4096: 0.95},
    }
    assert criteria["minimum_detections_at_n_600"] == {0.50: 321, 0.80: 497, 0.90: 553, 0.95: 580}
    assert criteria["maximum_false_positives_at_n_600"] == 20
    assert criteria["maximum_false_positive_rate"] == 0.05
    assert "one-sided exact" in criteria["acceptance"]
    assert "no exclusion" in criteria["abstentions"]


def test_preregistration_record_is_frozen_and_not_executed():
    record_path = Path(__file__).resolve().parents[1] / "reports/experiments/finite_shot_shot_aware_heldout_preregistration.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    frozen_hash = record["frozen_configuration_sha256"]
    record["frozen_configuration_sha256"] = None
    actual_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert frozen_hash == actual_hash
    assert record["status"] == "designed_not_executed"
    assert record["candidate_rule"]["cutoffs"] == {"256": 0.18359375, "1024": 0.09375, "4096": 0.0478515625}
