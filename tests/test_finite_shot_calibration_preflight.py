"""Static, no-simulation checks for the preregistered calibration plan."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# The repository config adds ``src`` to pytest's path but not the repository
# root, where this experiment-only plan module lives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.finite_shot_calibration_plan import (
    STRATA,
    _atomic_create_json,
    build_ledger,
    diagnosis_inputs,
    generate_candidate_template,
    generate_candidate_template_sets,
    generate_primary_template,
    generate_paired_case,
    generation_decisions,
    is_completed_payload,
    summarize_results,
    evaluate_preregistered_targets,
    validate_preflight,
    validate_execution_preflight,
)


def _namespace_ranges(spec):
    plan = spec["proposed_next_calibration_plan"]
    design = plan["design"]
    count = plan["trials_per_stratum_per_shot_per_split"]
    max_offset = 10000 * (len(STRATA) - 1) + 10 * (count - 1) + len(plan["shot_counts"]) - 1
    templates = design["template_generation_seed_roots"]
    sampling = design["sampling_seed_roots"]
    return {
        "calibration_template_generation": (templates["calibration"], templates["calibration"] + count - 1),
        "calibration_sampling": (sampling["calibration"], sampling["calibration"] + max_offset),
        "validation_template_generation": (templates["untouched_validation"], templates["untouched_validation"] + count - 1),
        "validation_sampling": (sampling["untouched_validation"], sampling["untouched_validation"] + max_offset),
    }


def test_seed_namespaces_are_explicit_and_pairwise_disjoint():
    spec_rows = validate_preflight()
    assert len(spec_rows) == 3600
    spec_ranges = _namespace_ranges(json.loads(
        (Path(__file__).parent / "fixtures" / "finite_shot_distribution_calibration.json").read_text()
    ))
    assert spec_ranges == {
        "calibration_template_generation": (20261014, 20261133),
        "calibration_sampling": (30261015, 30302207),
        "validation_template_generation": (40271014, 40271133),
        "validation_sampling": (50361015, 50402207),
    }
    ranges = list(spec_ranges.values())
    for i, (start_a, end_a) in enumerate(ranges):
        for start_b, end_b in ranges[i + 1:]:
            assert end_a < start_b or end_b < start_a


def test_ledger_workload_split_isolation_and_unique_ids():
    rows = build_ledger()
    assert len(rows) == 3600
    assert len({row["configuration_id"] for row in rows}) == len(rows)
    assert len({row["template_id"] for row in rows if row["split"] == "calibration"}) == 120
    assert len({row["template_id"] for row in rows if row["split"] == "untouched_validation"}) == 120
    calibration_templates = {row["template_id"] for row in rows if row["split"] == "calibration"}
    validation_templates = {row["template_id"] for row in rows if row["split"] == "untouched_validation"}
    calibration_seeds = {row["seed"] for row in rows if row["split"] == "calibration"}
    validation_seeds = {row["seed"] for row in rows if row["split"] == "untouched_validation"}
    assert calibration_templates.isdisjoint(validation_templates)
    assert calibration_seeds.isdisjoint(validation_seeds)
    assert all(row["output_location"].startswith(
        "reports/experiments/finite_shot_distribution_calibration_run/"
    ) for row in rows)
    assert sum(2 * row["shot_count"] for row in rows) == 12_902_400


def test_expected_labels_are_not_part_of_diagnosis_inputs():
    row = build_ledger()[0]
    case = {"submitted": [["h", 0]], "reference": [["h", 0]], "expected_label": "ANOMALY", "scenario": "subtle"}
    payload = diagnosis_inputs(case, row)
    assert set(payload) == {
        "circuit", "reference_circuit", "shots", "seed", "threshold",
    }
    assert "expected_label" not in payload
    assert "scenario" not in payload
    assert "fault_strength_stratum" not in payload


def test_interrupted_or_mismatched_outputs_are_not_counted_complete():
    row = build_ledger()[0]
    assert not is_completed_payload({"status": "running"}, row["configuration_id"])
    assert not is_completed_payload({
        "status": "completed", "configuration_id": row["configuration_id"],
    }, row["configuration_id"])
    assert not is_completed_payload({
        "status": "completed", "configuration_id": "wrong-id", "result": {"value": 1},
    }, row["configuration_id"])
    assert not is_completed_payload("{partial", row["configuration_id"])
    assert is_completed_payload({
        "status": "completed", "configuration_id": row["configuration_id"],
        "result": {"current_detected": False},
    }, row["configuration_id"])


def test_result_writer_is_atomic_and_never_overwrites():
    with tempfile.TemporaryDirectory(prefix=".atomic-result-test-", dir=Path(__file__).parent) as temp_dir:
        path = Path(temp_dir) / "nested" / "result.json"
        _atomic_create_json(path, {"status": "completed", "value": 1})
        assert json.loads(path.read_text()) == {"status": "completed", "value": 1}
        with pytest.raises(FileExistsError):
            _atomic_create_json(path, {"status": "completed", "value": 2})
        assert json.loads(path.read_text())["value"] == 1


def test_preflight_rejects_seed_namespace_overlap():
    from experiments.finite_shot_calibration_plan import load_spec

    spec = load_spec()
    spec["proposed_next_calibration_plan"]["design"]["sampling_seed_roots"]["calibration"] = 40271014
    with pytest.raises(ValueError, match="seed namespaces overlap"):
        validate_preflight(spec)


def test_primary_recipe_is_frozen_and_exploratory_topology_is_excluded():
    decision = generation_decisions()
    assert decision["status"] == "approved_and_frozen"
    assert decision["missing_decisions"] == []
    assert decision["primary_recipe"]["id"] == "fixed_pilot_topology_phase_tags"
    assert decision["primary_recipe"]["status"] == "selected_for_primary_calibration"
    assert decision["exploratory_recipes"][0]["id"] == "seeded_paired_topology_phase_tags"
    assert decision["exploratory_recipes"][0]["status"] == (
        "exploratory_only_outside_primary_preregistered_calibration"
    )
    rows = validate_execution_preflight()
    assert len(rows) == 3600
    template = generate_primary_template(20261014)
    assert template["recipe_id"] == "fixed_pilot_topology_phase_tags"
    assert template["generation_metadata"]["paired_cnot_edges"] == [[0, 2], [1, 3]]


def test_documented_pilot_pairs_and_fault_strengths_match_fixture_evidence():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "finite_shot_distribution_calibration.json").read_text())
    cases = {case["id"]: case for case in fixture["cases"]}
    identical = cases["same_circuit_control"]
    assert identical["reference"] == identical["submitted"]
    equivalent = cases["measurement_equivalent_phase_mutation"]
    assert equivalent["expected_anomaly"] is False
    assert equivalent["submitted"][:-2] == equivalent["reference"][:-1]
    assert equivalent["submitted"][-2] == ["z", 3]
    assert equivalent["submitted"][-1] == equivalent["reference"][-1]

    for case_id in ("subtle_ry_fault", "moderate_ry_fault", "large_ry_fault"):
        case = cases[case_id]
        original = case["reference"][0]
        intervention = case["submitted"][0]
        assert original[:2] == ["h", 0]
        assert intervention[:2] == ["ry", 0]
        assert abs(abs(__import__("math").cos(intervention[2])) / 2 - case["expected_ideal_tvd"]) < 0.00006
        assert case["expected_anomaly"] is True


def test_evaluator_reports_each_split_shot_stratum_and_exact_bounds():
    rows = [
        {"split": "calibration", "shot_count": 256, "scenario": "subtle", "fault_strength_stratum": "subtle", "expected_label": "ANOMALY", "current_detected": True, "permutation_detected": True},
        {"split": "calibration", "shot_count": 256, "scenario": "subtle", "fault_strength_stratum": "subtle", "expected_label": "ANOMALY", "current_detected": False, "permutation_detected": True},
        {"split": "calibration", "shot_count": 256, "scenario": "subtle", "fault_strength_stratum": "subtle", "expected_label": "NO_ANOMALY", "current_detected": False, "permutation_detected": True},
    ]
    summaries = summarize_results(rows)
    current = next(row for row in summaries if row["detector"] == "current_detected")
    permutation = next(row for row in summaries if row["detector"] == "permutation_detected")
    assert (current["tp"], current["fn"], current["fp"], current["tn"]) == (1, 1, 0, 1)
    assert current["sensitivity"] == current["recall"] == 0.5
    assert current["false_positive_rate"] == 0.0
    assert current["false_positive_upper_95_one_sided"] is not None
    assert (permutation["tp"], permutation["fp"]) == (2, 1)
    assert current["n_configurations"] == permutation["n_configurations"] == 3


def test_frozen_target_evaluator_uses_registered_counts_without_threshold_changes():
    summaries = [
        {"split": "calibration", "detector": "current_detected", "shot_count": 256,
         "scenario": "identical_control", "fp": 1},
        {"split": "calibration", "detector": "current_detected", "shot_count": 256,
         "scenario": "subtle", "tp": 70},
        {"split": "calibration", "detector": "current_detected", "shot_count": 1024,
         "scenario": "subtle", "tp": 104},
    ]
    outcomes = evaluate_preregistered_targets(summaries)
    assert [outcome["passed"] for outcome in outcomes] == [True, True, True]
    assert outcomes[0]["maximum"] == 1
    assert outcomes[1]["minimum"] == 70
    assert outcomes[2]["minimum"] == 104


@pytest.mark.parametrize("recipe_id", [
    "fixed_pilot_topology_phase_tags",
    "seeded_paired_topology_phase_tags",
])
def test_candidate_seed_sets_are_reproducible_distinct_and_split_isolated(recipe_id):
    one = generate_candidate_template(20261014, recipe_id)
    two = generate_candidate_template(20261014, recipe_id)
    assert one == two
    sets = generate_candidate_template_sets(recipe_id=recipe_id)
    calibration = sets["calibration"]
    validation = sets["untouched_validation"]
    assert len(calibration) == len(validation) == 120
    assert len({item["reference_fingerprint"] for item in calibration}) == 120
    assert len({item["reference_fingerprint"] for item in validation}) == 120
    assert {item["reference_fingerprint"] for item in calibration}.isdisjoint(
        {item["reference_fingerprint"] for item in validation}
    )
    assert generate_candidate_template(20261015, recipe_id)["reference_fingerprint"] != one["reference_fingerprint"]


def test_same_template_is_paired_across_all_strata_with_registered_interventions():
    rows = [row for row in build_ledger() if row["split"] == "calibration" and row["template_id"].endswith("000") and row["shot_count"] == 256]
    templates = generate_candidate_template_sets(recipe_id="fixed_pilot_topology_phase_tags")["calibration"]
    template = templates[0]
    cases = [generate_paired_case(template, row) for row in rows]
    assert {case["template_id"] for case in cases} == {rows[0]["template_id"]}
    assert len({case["reference_fingerprint"] for case in cases}) == 1
    by_stratum = {case["fault_strength_stratum"]: case for case in cases}
    assert by_stratum["identical_control"]["submitted"] == by_stratum["identical_control"]["reference"]
    equivalent = by_stratum["measurement_equivalent_control"]
    assert equivalent["submitted"][-2] == ("z", 3)
    assert equivalent["submitted"][:-2] == equivalent["reference"][:-1]
    for stratum, angle in (("subtle", 1.45), ("moderate", 1.2), ("large", 0.4)):
        case = by_stratum[stratum]
        changes = [(before, after) for before, after in zip(case["reference"], case["submitted"]) if before != after]
        assert changes == [(("h", 0), ("ry", 0, angle))]
        assert case["expected_label"] == "ANOMALY"


def test_candidate_templates_preserve_pilot_ideal_and_fixed_seed_sampling_behavior():
    from quantum_simulator.quantum_simulator import QuantumSimulator

    fixture = json.loads((Path(__file__).parent / "fixtures" / "finite_shot_distribution_calibration.json").read_text())
    pilot = {case["id"]: case for case in fixture["cases"]}
    pilot_ref = [tuple(operation) if isinstance(operation, list) else operation for operation in pilot["same_circuit_control"]["reference"]]
    for recipe_id in ("fixed_pilot_topology_phase_tags", "seeded_paired_topology_phase_tags"):
        template = generate_candidate_template(20261014, recipe_id)
        sim = QuantumSimulator(4, seed=13)
        for sample_seed in (13, 29, 47):
            candidate_counts = sim.run_shots(template["reference"], shots=256, seed=sample_seed)
            pilot_counts = QuantumSimulator(4, seed=sample_seed).run_shots(pilot_ref, shots=256, seed=sample_seed)
            assert candidate_counts == pilot_counts
        for stratum, case_id in (("subtle", "subtle_ry_fault"), ("moderate", "moderate_ry_fault"), ("large", "large_ry_fault")):
            paired = generate_paired_case(template, next(
                row for row in build_ledger()
                if row["split"] == "calibration" and row["template_id"].endswith("000")
                and row["fault_strength_stratum"] == stratum and row["shot_count"] == 256
            ))
            ref_sim = QuantumSimulator(4, seed=13)
            ref_sim.run_circuit(paired["reference"][:-1])
            reference_probs = ref_sim.get_probabilities()
            obs_sim = QuantumSimulator(4, seed=13)
            obs_sim.run_circuit(paired["submitted"][:-1])
            observed_probs = obs_sim.get_probabilities()
            ideal_tvd = 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(reference_probs, observed_probs))
            assert round(ideal_tvd, 4) == pilot[case_id]["expected_ideal_tvd"]
            pilot_obs = [tuple(operation) if isinstance(operation, list) else operation for operation in pilot[case_id]["submitted"][:-1]]
            pilot_sim = QuantumSimulator(4, seed=13)
            pilot_sim.run_circuit(pilot_obs)
            # A bitstring relabeling may permute category labels. The sampled
            # TVD/permutation statistic is label-invariant, so matching paired
            # PMFs establish the same finite-shot law for these statistics.
            assert sorted(float(p) for p in observed_probs) == pytest.approx(
                sorted(float(p) for p in pilot_sim.get_probabilities()), abs=1e-12
            )


def test_unsupported_candidate_recipe_and_stratum_fail_clearly():
    with pytest.raises(ValueError, match="unsupported calibration template recipe"):
        generate_candidate_template(7, "unregistered")
    row = dict(build_ledger()[0], fault_strength_stratum="unknown")
    template = generate_candidate_template(row["template_seed"], "fixed_pilot_topology_phase_tags")
    with pytest.raises(ValueError, match="unsupported calibration stratum"):
        generate_paired_case(template, row)


def test_bounded_recipe_comparison_120_templates_and_finite_shot_subset(capsys):
    """Compare both recipes without using the untouched validation split.

    Exact statevector metrics cover all 120 calibration templates. Finite-shot
    sampling uses a fixed, comparison-only seed namespace and template indices
    0, 59, and 119 at every registered shot count and stratum.
    """
    import hashlib
    import json
    import numpy as np

    from quantum_simulator.quantum_simulator import QuantumSimulator
    from quantum_simulator.statistics import compare_distributions

    spec = json.loads((Path(__file__).parent / "fixtures" / "finite_shot_distribution_calibration.json").read_text())
    plan = spec["proposed_next_calibration_plan"]
    count = int(plan["trials_per_stratum_per_shot_per_split"])
    template_root = int(plan["design"]["template_generation_seed_roots"]["calibration"])
    shots_grid = tuple(plan["shot_counts"])
    finite_indices = (0, 59, 119)
    comparison_seed_root = 60_000_000
    strata = ("identical_control", "measurement_equivalent_control", "subtle", "moderate", "large")
    angles = {"subtle": 1.45, "moderate": 1.2, "large": 0.4}
    ideal_rows = []
    finite_rows = []
    topology_fingerprints = {}
    ledger_rows = build_ledger()
    comparison_rows = {
        (row["template_id"], row["fault_strength_stratum"]): row
        for row in ledger_rows
        if row["split"] == "calibration" and row["shot_count"] == shots_grid[0]
    }

    def probabilities(instructions):
        sim = QuantumSimulator(4, seed=1)
        sim.run_circuit(instructions[:-1])
        values = sim.get_probabilities()
        return {f"{index:04b}": float(value) for index, value in enumerate(values)}

    def distribution(counts):
        total = sum(counts.values())
        return {f"{index:04b}": counts.get(f"{index:04b}", 0) / total for index in range(16)}

    for recipe_id in ("fixed_pilot_topology_phase_tags", "seeded_paired_topology_phase_tags"):
        templates = [generate_candidate_template(template_root + index, recipe_id) for index in range(count)]
        topology_fingerprints[recipe_id] = {
            hashlib.sha256(json.dumps(item["generation_metadata"]["paired_cnot_edges"]).encode()).hexdigest()
            for item in templates
        }
        for template_index, template in enumerate(templates):
            reference = template["reference"]
            for stratum_index, stratum in enumerate(strata):
                template_id = f"calibration:template:{template_index:03d}"
                row = comparison_rows[(template_id, stratum)]
                case = generate_paired_case(template, row)
                expected = probabilities(reference)
                observed = probabilities(case["submitted"])
                ideal_tvd = 0.5 * sum(abs(expected[key] - observed[key]) for key in expected)
                ideal_rows.append({"recipe": recipe_id, "template": template_index,
                                   "stratum": stratum, "tvd": ideal_tvd})
                if template_index not in finite_indices:
                    continue
                for shot_index, shot_count in enumerate(shots_grid):
                    sample_seed = comparison_seed_root + template_index * 1000 + stratum_index * 10 + shot_index * 2
                    ref_counts = QuantumSimulator(4, seed=sample_seed).run_shots(
                        reference, shots=shot_count, seed=sample_seed)
                    obs_counts = QuantumSimulator(4, seed=sample_seed + 1).run_shots(
                        case["submitted"], shots=shot_count, seed=sample_seed + 1)
                    result = compare_distributions(distribution(ref_counts), distribution(obs_counts),
                                                   threshold=0.05, shots=shot_count)
                    finite_rows.append({"recipe": recipe_id, "template": template_index,
                                        "stratum": stratum, "shots": shot_count,
                                        "tvd": result.total_variation_distance,
                                        "detected": result.anomaly_detected})

    assert len(ideal_rows) == 2 * count * len(strata)
    assert len(finite_rows) == 2 * len(finite_indices) * len(strata) * len(shots_grid)
    assert len(topology_fingerprints["fixed_pilot_topology_phase_tags"]) == 1
    assert len(topology_fingerprints["seeded_paired_topology_phase_tags"]) > 1
    ideal_summary = {}
    for recipe_id in ("fixed_pilot_topology_phase_tags", "seeded_paired_topology_phase_tags"):
        ideal_summary[recipe_id] = {}
        for stratum in strata:
            values = [row["tvd"] for row in ideal_rows if row["recipe"] == recipe_id and row["stratum"] == stratum]
            ideal_summary[recipe_id][stratum] = {
                "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))
            }
            if stratum in angles:
                assert np.allclose(values, abs(np.cos(angles[stratum])) / 2, atol=1e-12)
            else:
                assert max(values) < 1e-12
    finite_summary = {}
    for recipe_id in ("fixed_pilot_topology_phase_tags", "seeded_paired_topology_phase_tags"):
        for shot_count in shots_grid:
            for stratum in strata:
                values = [row for row in finite_rows if row["recipe"] == recipe_id
                          and row["shots"] == shot_count and row["stratum"] == stratum]
                finite_summary[f"{recipe_id}|{shot_count}|{stratum}"] = {
                    "detected": sum(row["detected"] for row in values), "n": len(values),
                    "mean_tvd": float(np.mean([row["tvd"] for row in values]))
                }
    print("RECIPE_COMPARISON=" + json.dumps({
        "ideal": ideal_summary,
        "finite_shot": finite_summary,
        "topology_counts": {key: len(value) for key, value in topology_fingerprints.items()},
        "finite_templates": list(finite_indices),
        "comparison_sampling_seed_root": comparison_seed_root,
        "shot_circuit_executions": 2 * len(finite_indices) * len(strata) * 2 * sum(shots_grid),
    }, sort_keys=True))
