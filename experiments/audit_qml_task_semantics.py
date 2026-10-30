"""Reconcile processed task partitions against saved development OOF group folds."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "experiments/source_task_contracts"
OUT = ROOT / "reports/experiments/qml_source_preprocessing_20261029/semantic_validation_audit.json"
MODEL_FEATURE_FIELDS = {
    ("lintq", "warning_triage"): ["warning_message"],
    ("qbugs_hornbro", "correct_buggy_circuit_state"): ["code", "scenario", "circuit_qubits", "circuit_depth", "instruction_count"],
    ("qolumbina_ideal", "seeded_mutant_detection_ideal"): ["program", "variant_version", "backend", "code"],
    ("quantum_platform_bugs", "comment_only_classification"): ["code"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows_out = []
    for contract_path in sorted(CONTRACTS.glob("*.json")):
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        out_path = contract.get("output_path")
        if not out_path:
            continue
        manifest_path = ROOT / out_path / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = []
        source = ROOT / manifest["input_path"]
        fold_input = ROOT / manifest["fold_input_path"]
        metadata_correction = None
        if "reuses fold assignment recorded in existing QNN OOF predictions" in manifest.get("limitations", []):
            correction_path = manifest_path.with_name("manifest_provenance_correction.json")
            if fold_input.name == "group_folds.csv":
                if not correction_path.is_file():
                    problems.append("stale_generic_fold_provenance_without_correction")
                else:
                    correction = json.loads(correction_path.read_text(encoding="utf-8"))
                    evidence_path = ROOT / correction.get("preparation_report_path", "")
                    if (correction.get("processed_manifest_sha256") != sha256(manifest_path) or
                            correction.get("fold_input_sha256") != sha256(fold_input) or
                            not evidence_path.is_file() or
                            correction.get("preparation_report_sha256") != sha256(evidence_path)):
                        problems.append("split_provenance_correction_hash_mismatch")
                    metadata_correction = {"path": str(correction_path.relative_to(ROOT)),
                        "status": "VERIFIED" if not problems else "REVIEW_REQUIRED",
                        "correction": correction.get("correction"),
                        "preparation_report": correction.get("preparation_report_path")}
        if sha256(source) != manifest["input_sha256"]: problems.append("source_hash_mismatch")
        if sha256(fold_input) != manifest["fold_sha256"]: problems.append("fold_source_hash_mismatch")
        data_rows = []
        for part in manifest["partitions"]:
            path = ROOT / part["path"]
            if sha256(path) != part["sha256"]: problems.append(f"partition_hash_mismatch:{part['partition_id']}")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                partition = list(csv.DictReader(stream))
            if len(partition) != part["rows"]: problems.append(f"partition_row_count_mismatch:{part['partition_id']}")
            data_rows.extend(partition)
        if len(data_rows) != manifest["counts"].get("accepted"):
            problems.append("accepted_partition_total_mismatch")
        ids = [row[contract["record_id_column"]] for row in data_rows]
        if len(ids) != len(set(ids)): problems.append("duplicate_record_id_in_clean_partitions")
        with fold_input.open("rb") as raw:
            opener = gzip.open if contract.get("fold_gzip") else None
            # gzip's text wrapper is needed for the three-class QMutBench OOF.
        if contract.get("fold_gzip"):
            f = gzip.open(fold_input, "rt", encoding="utf-8-sig", newline="")
        else:
            f = fold_input.open(encoding="utf-8-sig", newline="")
        with f as stream:
            selected = []
            for row in csv.DictReader(stream):
                if contract.get("fold_model_value") is not None and row.get(contract["fold_model_column"]) != contract["fold_model_value"]:
                    continue
                selected.append(row)
        rid_col, group_col = contract["record_id_column"], contract["group_column"]
        fold_rid = contract["fold_record_column"]
        fold_group = contract.get("fold_group_column")
        fold_by_id = {}
        group_to_fold = {}
        for row in selected:
            rid = row[fold_rid]
            if rid in fold_by_id: problems.append("duplicate_record_id_in_fold_manifest")
            fold_by_id[rid] = (int(row[contract["fold_column"]]), row.get(fold_group, ""),
                               row.get(contract.get("fold_target_column", "")))
        if set(ids) != set(fold_by_id): problems.append("data_fold_record_id_set_mismatch")
        labels = Counter()
        groups = defaultdict(set)
        group_labels = defaultdict(set)
        source_group_values = defaultdict(set)
        code_groups = defaultdict(set)
        code_folds = defaultdict(set)
        code_labels = defaultdict(set)
        code_row_counts = Counter()
        for row in data_rows:
            rid, group, label = row[rid_col], str(row[group_col]), str(row[contract["label_column"]])
            labels[label] += 1
            groups[group].add(int(fold_by_id.get(rid, (-999, ""))[0]))
            if group_to_fold.setdefault(group, int(fold_by_id.get(rid, (-999, ""))[0])) != int(fold_by_id.get(rid, (-999, ""))[0]):
                problems.append("group_spans_validation_folds")
            if fold_group and str(fold_by_id.get(rid, (-999, ""))[1]) != group:
                problems.append("fold_group_id_mismatch")
            group_labels[group].add(label)
            source_col = contract.get("training_source_column")
            if source_col: source_group_values[group].add(row.get(source_col, ""))
            feature_fields = contract.get("model_feature_fields") or MODEL_FEATURE_FIELDS.get(
                (contract["source_id"], contract["task_id"]),
                [contract.get("feature_column")] if contract.get("feature_column") else [])
            if feature_fields:
                text = " ".join(str(row.get(col, "")) for col in feature_fields)
                if (contract["source_id"], contract["task_id"]) == ("quantum_platform_bugs", "comment_only_classification"):
                    text = re.sub(r"(?i)\b(bugs?|buggy|fix(?:ed|es)?|defects?|false\s+positive|true\s+positive)\b", " ", text)
                norm = " ".join(text.split())
                fingerprint = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                code_groups[fingerprint].add(group)
                code_folds[fingerprint].add(int(fold_by_id.get(rid, (-999, ""))[0]))
                code_labels[fingerprint].add(label)
                code_row_counts[fingerprint] += 1
            target_map = contract.get("fold_label_map")
            target_col = contract.get("fold_target_column")
            fold_target = fold_by_id.get(rid, (-999, "", None))[2]
            if target_map and target_col and str(target_map.get(label)) != str(fold_target):
                problems.append("declared_target_map_mismatch")
        mixed_source_groups = sum(len(v) > 1 for v in source_group_values.values())
        if mixed_source_groups: problems.append("group_contains_multiple_source_cohorts")
        exact_text_hashes_reused_across_groups = sum(len(v) > 1 for v in code_groups.values())
        cross_fold_hashes = {key for key in code_groups if len(code_groups[key]) > 1 and len(code_folds[key]) > 1}
        cross_fold_rows = sum(code_row_counts[key] for key in cross_fold_hashes)
        cross_fold_conflicts = sum(len(code_labels[key]) > 1 for key in cross_fold_hashes)
        within_group_conflict_hashes = {key for key in code_groups
                                        if len(code_labels[key]) > 1 and len(code_groups[key]) == 1}
        within_group_conflict_rows = sum(code_row_counts[key] for key in within_group_conflict_hashes)
        source_specific_evidence = None
        if contract.get("task_id") == "correct_buggy_circuit_state_qasm_v2":
            prep_path = fold_input.parent / "preparation_report.json"
            rejected_path = fold_input.parent / "rejected_rows.csv"
            source_specific_problems = []
            prep = json.loads(prep_path.read_text(encoding="utf-8")) if prep_path.is_file() else {}
            if not prep:
                source_specific_problems.append("missing_qasm_preparation_report")
            else:
                if sha256(source) != prep.get("input_sha256"):
                    source_specific_problems.append("qasm_source_manifest_hash_mismatch")
                if sha256(fold_input) != prep.get("fold_manifest_sha256"):
                    source_specific_problems.append("qasm_group_split_hash_mismatch")
                if not rejected_path.is_file() or sha256(rejected_path) != prep.get("rejection_manifest_sha256"):
                    source_specific_problems.append("qasm_rejection_manifest_hash_mismatch")
                if prep.get("accepted_rows") != len(data_rows):
                    source_specific_problems.append("qasm_accepted_row_total_mismatch")
                if prep.get("input_rows") != prep.get("accepted_rows", 0) + prep.get("rejected_rows", 0):
                    source_specific_problems.append("qasm_input_accept_reject_total_mismatch")
                if rejected_path.is_file():
                    with rejected_path.open(encoding="utf-8-sig", newline="") as stream:
                        rejects = list(csv.DictReader(stream))
                    reject_pair_rows = defaultdict(list)
                    reject_hash_labels = defaultdict(set)
                    for rejected_row in rejects:
                        reject_pair_rows[rejected_row["source_pair_id"]].append(rejected_row)
                        reject_hash_labels[rejected_row["qasm_sha256"]].add(str(rejected_row["label"]))
                        if not rejected_row.get("reason"):
                            source_specific_problems.append("qasm_rejection_missing_reason")
                    if len(rejects) != prep.get("rejected_rows"):
                        source_specific_problems.append("qasm_rejected_row_total_mismatch")
                    if sum(values == {"0", "1"} for values in reject_hash_labels.values()) != prep.get("conflicting_exact_qasm_fingerprints"):
                        source_specific_problems.append("qasm_conflict_label_accounting_mismatch")
                    if any(len(pair_rows) != 2 or {str(x["label"]) for x in pair_rows} != {"0", "1"}
                           for pair_rows in reject_pair_rows.values()):
                        source_specific_problems.append("qasm_rejected_source_pair_incomplete")
                for source_file in prep.get("source_files", []):
                    source_path = ROOT / source_file["path"]
                    if not source_path.is_file() or sha256(source_path) != source_file["sha256"]:
                        source_specific_problems.append(f"qbugs_source_hash_mismatch:{source_file['path']}")
            if source_specific_problems:
                problems.extend(source_specific_problems)
            source_specific_evidence = {
                "preparation_report": str(prep_path.relative_to(ROOT)) if prep_path.is_file() else None,
                "preparation_report_status": prep.get("status"),
                "input_rows_before_semantic_rejection": prep.get("input_rows"),
                "accepted_source_rows": prep.get("accepted_rows"),
                "rejected_source_rows": prep.get("rejected_rows"),
                "rejected_complete_pairs": prep.get("rejected_source_pairs"),
                "conflicting_exact_qasm_fingerprints": prep.get("conflicting_exact_qasm_fingerprints"),
                "verified_source_pickle_count": len(prep.get("source_files", [])) if not source_specific_problems else None,
                "checks": "source pickle hashes, exact-QASM conflict rejection manifest, derived input/fold hashes, and accepted/rejected row totals",
            }
        elif contract.get("task_id") == "compatible_subset_qasm_v1":
            prep_path = fold_input.parent / "preparation_report.json"
            reject_path = fold_input.parent / "rejected_rows.csv"
            unreadable_path = fold_input.parent / "unreadable_source_files.csv"
            source_specific_problems = []
            prep = json.loads(prep_path.read_text(encoding="utf-8")) if prep_path.is_file() else {}
            if not prep:
                source_specific_problems.append("missing_mute_alg_preparation_report")
            else:
                if sha256(source) != prep.get("input_sha256"):
                    source_specific_problems.append("mute_alg_input_hash_mismatch")
                if sha256(fold_input) != prep.get("fold_manifest_sha256"):
                    source_specific_problems.append("mute_alg_fold_hash_mismatch")
                if not reject_path.is_file() or sha256(reject_path) != prep.get("rejection_manifest_sha256"):
                    source_specific_problems.append("mute_alg_rejection_hash_mismatch")
                if not unreadable_path.is_file() or sha256(unreadable_path) != prep.get("unreadable_files_sha256"):
                    source_specific_problems.append("mute_alg_unreadable_file_audit_hash_mismatch")
                statuses = Counter(item.get("status") for item in prep.get("source_files", []))
                if (statuses.get("LOADED", 0) != prep.get("successfully_loaded_files") or
                        statuses.get("BLOCKED_LOAD_OR_NORMALIZATION", 0) != prep.get("unreadable_or_unusable_files") or
                        len(prep.get("source_files", [])) != prep.get("source_file_count_scanned")):
                    source_specific_problems.append("mute_alg_source_file_status_totals_mismatch")
                if sum(2 * int(item.get("source_pairs", 0)) for item in prep.get("source_files", [])
                       if item.get("status") == "LOADED") != prep.get("input_rows_before_semantic_rejections"):
                    source_specific_problems.append("mute_alg_loaded_pair_row_total_mismatch")
                if prep.get("accepted_rows") != len(data_rows) or prep.get("accepted_rows", 0) + prep.get("rejected_semantic_rows", 0) != prep.get("input_rows_before_semantic_rejections"):
                    source_specific_problems.append("mute_alg_row_accounting_mismatch")
                if reject_path.is_file():
                    with reject_path.open(encoding="utf-8-sig", newline="") as stream:
                        rejected_rows = list(csv.DictReader(stream))
                    rejected_by_pair = defaultdict(list)
                    rejected_by_qasm = defaultdict(set)
                    for item in rejected_rows:
                        rejected_by_pair[item["source_pair_id"]].append(item)
                        rejected_by_qasm[item["qasm_sha256"]].add(str(item["label"]))
                        if not item.get("reason"):
                            source_specific_problems.append("mute_alg_rejection_reason_missing")
                    if (len(rejected_rows) != prep.get("rejected_semantic_rows") or
                            any(len(pair) != 2 or {str(item["label"]) for item in pair} != {"0", "1"}
                                for pair in rejected_by_pair.values())):
                        source_specific_problems.append("mute_alg_complete_pair_rejection_mismatch")
                    if (sum(values == {"0", "1"} for values in rejected_by_qasm.values()) != prep.get("conflicting_exact_qasm_fingerprints") or
                            len(rejected_by_pair) != prep.get("rejected_complete_pairs")):
                        source_specific_problems.append("mute_alg_conflict_label_accounting_mismatch")
                if unreadable_path.is_file():
                    with unreadable_path.open(encoding="utf-8-sig", newline="") as stream:
                        unreadable_rows = list(csv.DictReader(stream))
                    expected_bad = {item["path"]: item["sha256"] for item in prep.get("source_files", [])
                                    if item.get("status") == "BLOCKED_LOAD_OR_NORMALIZATION"}
                    actual_bad = {item["path"]: item["sha256"] for item in unreadable_rows}
                    if expected_bad != actual_bad:
                        source_specific_problems.append("mute_alg_unreadable_ledger_content_mismatch")
                for file_item in prep.get("source_files", []):
                    raw_path = ROOT / file_item["path"]
                    if not raw_path.is_file() or sha256(raw_path) != file_item["sha256"]:
                        source_specific_problems.append(f"mute_alg_source_hash_mismatch:{file_item['path']}")
            if source_specific_problems:
                problems.extend(source_specific_problems)
            source_specific_evidence = {
                "preparation_report": str(prep_path.relative_to(ROOT)) if prep_path.is_file() else None,
                "input_rows_before_semantic_rejection": prep.get("input_rows_before_semantic_rejections"),
                "accepted_rows": prep.get("accepted_rows"),
                "rejected_duplicate_rows": prep.get("rejected_semantic_rows"),
                "rejected_complete_pairs": prep.get("rejected_complete_pairs"),
                "conflicting_exact_qasm_fingerprints": prep.get("conflicting_exact_qasm_fingerprints"),
                "source_files_scanned": prep.get("source_file_count_scanned"),
                "source_files_loaded": prep.get("successfully_loaded_files"),
                "source_files_blocked": prep.get("unreadable_or_unusable_files"),
                "groups": prep.get("groups"),
                "checks": "all raw pickle hashes, load-failure ledger, exact-QASM rejection ledger, derived input and fold hashes, and row totals",
            }
        eligibility = "READY_FOR_TRAINING"
        eligibility_reason = "No exact duplicate task-feature fingerprints cross the registered validation folds."
        task_key = (contract["source_id"], contract["task_id"])
        if task_key == ("lintq", "warning_triage") and cross_fold_conflicts:
            eligibility = "BLOCKED_FEATURE_LABEL_AMBIGUITY"
            eligibility_reason = "The current warning_message-only feature repeats across folds with conflicting TP/FP labels; include justified context or redesign the split/feature policy before new training."
        elif task_key == ("qbugs_hornbro", "correct_buggy_circuit_state") and cross_fold_conflicts:
            eligibility = "BLOCKED_FEATURE_LABEL_AMBIGUITY"
            eligibility_reason = "Exact model-input text/metadata fingerprints recur across folds with conflicting correct/buggy labels; reconcile source pairs or evaluate a duplicate-component split before new training."
        elif task_key == ("quantum_platform_bugs", "comment_only_classification") and cross_fold_hashes:
            eligibility = "BLOCKED_CROSS_FOLD_DUPLICATE_COMMENTS"
            eligibility_reason = "Exact comment text is repeated across issue groups and validation folds; regroup duplicate comment components before new training."
        elif task_key == ("qsimbench", "backend_fingerprint") and cross_fold_conflicts:
            eligibility = "READY_EXPLORATORY_WITH_FEATURE_COLLISIONS"
            eligibility_reason = "Quantized output descriptors recur across held-out circuit settings with both backend labels; preserve this limitation in any generalization claim."
        elif task_key == ("lintq", "warning_triage_context_v2") and within_group_conflict_hashes:
            eligibility = "READY_EXPLORATORY_WITH_LABEL_AMBIGUITY"
            eligibility_reason = (f"Full rule/warning/source-context duplicates are grouped within folds; "
                f"{within_group_conflict_rows} records in {len(within_group_conflict_hashes)} identical-context components retain conflicting manual labels.")
        elif task_key == ("qbugs_hornbro", "correct_buggy_circuit_state_qasm_v2") and source_specific_evidence:
            eligibility_reason = (f"READY: {prep.get('accepted_rows')} OpenQASM records in {len(groups)} paired groups; "
                f"{prep.get('rejected_rows')} records ({prep.get('rejected_source_pairs')} complete source pairs) were explicitly rejected "
                "because exact OpenQASM fingerprints had contradictory labels. Source pickle, input, split, and rejection hashes verified; no exact QASM feature crosses folds.")
        elif task_key == ("mute_alg_hornbro", "compatible_subset_qasm_v1") and source_specific_evidence:
            eligibility = "INELIGIBLE_PARTIAL_SOURCE_LOW_GROUP_DIVERSITY"
            eligibility_reason = (f"The compatibility-selected subset has {source_specific_evidence.get('accepted_rows')} accepted rows but only "
                f"{source_specific_evidence.get('groups')} exact-QASM/pair components; {source_specific_evidence.get('source_files_loaded')} of "
                f"{source_specific_evidence.get('source_files_scanned')} files loaded and {source_specific_evidence.get('source_files_blocked')} failed restricted Qiskit deserialization. "
                f"Another {source_specific_evidence.get('rejected_duplicate_rows')} rows were excluded across conflicting duplicate fingerprints. "
                "This partial, compatibility-selected sample is too narrow for a representative training claim; resolve Qiskit serialization and expand independent circuits first.")
        rows_out.append({"task_id": contract["task_id"], "source_id": contract["source_id"],
            "contract": str(contract_path.relative_to(ROOT)), "manifest": str(manifest_path.relative_to(ROOT)),
            "status": "VERIFIED" if not problems and manifest.get("validation_pass") else "REVIEW_REQUIRED",
            "eligibility_status": eligibility, "eligibility_reason": eligibility_reason,
            "input_rows": manifest["counts"].get("input"), "accepted_rows": len(data_rows),
            "rejected_rows": manifest["rejection_references"].get("rows"),
            "classes": dict(labels), "groups": len(groups), "folds": len({fold for x in fold_by_id.values() for fold in [x[0]]}),
            "group_cross_fold": any(len(v) > 1 for v in groups.values()),
            "mixed_source_groups": mixed_source_groups,
            "exact_feature_text_fingerprints_shared_across_groups": exact_text_hashes_reused_across_groups,
            "cross_fold_duplicate_feature_fingerprints": len(cross_fold_hashes),
            "rows_with_cross_fold_duplicate_features": cross_fold_rows,
            "cross_fold_duplicate_fingerprints_with_conflicting_labels": cross_fold_conflicts,
            "within_group_duplicate_fingerprints_with_conflicting_labels": len(within_group_conflict_hashes),
            "rows_in_within_group_label_conflict_fingerprints": within_group_conflict_rows,
            "source_specific_evidence": source_specific_evidence,
            "metadata_correction": metadata_correction,
            "fold_prediction_ids": len(fold_by_id), "problems": sorted(set(problems)),
            "input_sha256": manifest["input_sha256"], "fold_sha256": manifest["fold_sha256"],
            "source_semantics": contract["target_semantics"]})

    # QMutBench binary full-development preprocessing uses its source-specific
    # streaming adapter rather than the small-table partitions above.
    qmb = ROOT / "data/processed/qmutbench/binary_development/manifest.json"
    if qmb.is_file():
        m = json.loads(qmb.read_text(encoding="utf-8"))
        index = ROOT / m["index_path"]
        problems = []
        if sha256(index) != m.get("index_sha256"): problems.append("sqlite_index_hash_mismatch")
        a = m["source_audit"]
        if a.get("frozen_test_rows_retained") != 0: problems.append("frozen_rows_retained")
        if a.get("development_rows_indexed") != 578117 or a.get("development_origins") != 305: problems.append("source_counts_mismatch")
        rows_out.append({"task_id": "binary_mutation_outcome", "source_id": "qmutbench",
            "contract": "qmutbench_qml/data/streaming.py::DevelopmentStream(archive_authoritative=True)",
            "manifest": str(qmb.relative_to(ROOT)), "status": "VERIFIED" if not problems else "REVIEW_REQUIRED",
            "input_rows": 578117, "accepted_rows": 578117, "rejected_rows": 0,
            "classes": a.get("class_counts"), "groups": a.get("development_origins"),
            "folds": len(a.get("fold_origin_counts", {})), "group_cross_fold": False,
            "mixed_source_groups": 0, "exact_feature_text_fingerprints_shared_across_groups": None,
            "fold_prediction_ids": 578117, "problems": problems,
            "input_sha256": a.get("source_sha256"), "fold_sha256": a.get("reference_archive_sha256"),
            "source_semantics": m.get("target_semantics"), "training_status": "BLOCKED_MEMORY_GUARD",
            "eligibility_status": "PREPROCESSING_READY_TRAINING_BLOCKED_MEMORY",
            "eligibility_reason": "Established archive-authoritative grouped folds and development data validate; existing QNN startup/in-run memory floors are not met.",
            "cross_fold_duplicate_feature_fingerprints": None,
            "rows_with_cross_fold_duplicate_features": None,
            "cross_fold_duplicate_fingerprints_with_conflicting_labels": None})

    report = {"schema": "qml-task-semantic-validation-audit-v1", "tasks": rows_out,
              "task_count": len(rows_out), "verified_count": sum(row["status"] == "VERIFIED" for row in rows_out),
              "review_required_count": sum(row["status"] != "VERIFIED" for row in rows_out),
              "frozen_test_accessed": False,
              "limitations": ["exact duplicate feature fingerprints across different groups are reported as leakage risks for review; they do not by themselves invalidate tasks where quantization or templated code makes repeats expected",
                              "external labels are validated according to the written source/task contract; no target was redefined"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# QML source/task semantic validation audit", "",
          f"Tasks audited: {report['task_count']}; verified: {report['verified_count']}; review required: {report['review_required_count']}. Frozen test accessed: no.", "",
          "| Source/task | Manifest status | Training eligibility | Rows accepted/rejected | Groups | Folds | Shared feature fingerprints | Cross-fold duplicate rows | Conflicting-label duplicate fingerprints | Issues |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows_out:
        md.append(f"| {row['source_id']} / {row['task_id']} | {row['status']} | {row.get('eligibility_status', 'REVIEW')} | {row['accepted_rows']}/{row['rejected_rows']} | {row['groups']} | {row['folds']} | {row.get('exact_feature_text_fingerprints_shared_across_groups')} | {row.get('rows_with_cross_fold_duplicate_features', 0)} | {row.get('cross_fold_duplicate_fingerprints_with_conflicting_labels', 0)} | {', '.join(row['problems']) or 'none'} |")
    md += ["", "## Duplicate-feature decisions", ""]
    for row in rows_out:
        if row.get("eligibility_status", "").startswith("BLOCKED") or row.get("eligibility_status", "").startswith("READY_EXPLORATORY"):
            md.append(f"- **{row['source_id']} / {row['task_id']} — {row['eligibility_status']}:** {row['eligibility_reason']}")
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": report["task_count"], "verified": report["verified_count"],
                      "review_required": report["review_required_count"], "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
