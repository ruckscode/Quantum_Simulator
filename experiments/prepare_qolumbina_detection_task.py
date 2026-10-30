"""Build a program-grouped mutant detectability task from Qolumbina reports.

One sample is a documented mutant/backend pair. Its target is whether the
published DOSS reports detected it in a majority of available random-seed runs.
Only source code and the pre-execution backend are features; report outputs are
never included as model input.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_quantum_datasets/qolumbina/verified_extracted/qolumbina-empirical-study-main/analysis/rq3"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection"


def extended(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" and not value.startswith("\\\\?\\") else value


def strip_comments_and_docstrings(source: str) -> str:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
        spans = []
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                    spans.append((first.lineno, getattr(first, "end_lineno", first.lineno)))
        for a, b in spans:
            for i in range(a - 1, b):
                lines[i] = ""
    except SyntaxError:
        pass
    without_docs = "\n".join(lines)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(without_docs).readline)
        tokens = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        return tokenize.untokenize(tokens)
    except (tokenize.TokenError, IndentationError):
        return without_docs


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    root = extended(DATA)
    variant_root = os.path.join(root, "buggy_variants")
    source_map = {}
    for base, dirs, files in os.walk(variant_root):
        for name in files:
            m = re.fullmatch(r"(.+)_v([1-7])\.py", name, re.IGNORECASE)
            if m:
                source_map[(m.group(1).casefold(), int(m.group(2)))] = os.path.join(base, name)

    reports_root = os.path.join(root, "data", "effectiveness", "test_reports")
    grouped = defaultdict(list)
    raw_report_count = 0
    ignored_bugless = 0
    status_counts = Counter()
    for base, dirs, files in os.walk(reports_root):
        # `_local_check` files are smoke/test fixtures, not published observations.
        dirs[:] = [d for d in dirs if d != "_local_check"]
        for name in files:
            if not name.lower().endswith(".json"):
                continue
            report = load_json(os.path.join(base, name))
            version = str(report.get("version", ""))
            if version == "bugless":
                ignored_bugless += 1
                continue
            match = re.search(r"_buggy_v([1-7])_\d+\.json$", name, re.IGNORECASE)
            if not match:
                raise ValueError(f"Unexpected Qolumbina report name: {name}")
            program = str(report["program_name"])
            ver = int(match.group(1))
            # The report uses `deutsch_jozsa`; its source file stem is `dj`.
            candidate_keys = [(program.casefold(), ver)]
            if program.casefold() == "deutsch_jozsa":
                candidate_keys.append(("dj", ver))
            source_path = next((source_map[k] for k in candidate_keys if k in source_map), None)
            if source_path is None:
                raise FileNotFoundError(f"No marked variant source for {program} v{ver}")
            cfg = report.get("test_configuration", {})
            backend = str(cfg.get("backend", "unknown"))
            seed = int(cfg.get("random_seed", -1))
            results = [x.get("test_result") for x in report.get("test_report", {}).values() if isinstance(x, dict)]
            results = [x for x in results if x in {"pass", "fail"}]
            if not results:
                raise ValueError(f"No valid published oracle results in {name}")
            detected = int("fail" in results)
            status_counts.update(results)
            raw_report_count += 1
            grouped[(program, ver, backend)].append({"seed": seed, "detected": detected,
                "test_count": len(results), "source_path": source_path})

    programs = sorted({k[0] for k in grouped})
    program_ids = {name: i + 1 for i, name in enumerate(programs)}
    rows = []
    missing_seed_keys = []
    for (program, ver, backend), reports in sorted(grouped.items()):
        unique_seeds = {x["seed"] for x in reports}
        rates = sum(x["detected"] for x in reports) / len(reports)
        source_path = reports[0]["source_path"]
        with open(source_path, encoding="utf-8") as f:
            source = strip_comments_and_docstrings(f.read())
        if len({x["source_path"] for x in reports}) != 1:
            raise ValueError("One program/version/backend maps to multiple variant files")
        # The backend is available before a run. Test outputs and oracle results
        # are strictly withheld from features.
        feature_text = source + f"\nbackend_{backend.casefold()}"
        rows.append({"record_id": f"{program}:v{ver}:{backend}", "bug_id": program_ids[program],
            "program": program, "variant_version": ver, "backend": backend,
            "label": int(rates >= 0.5), "detection_rate": rates, "runs": len(reports),
            "unique_random_seeds": len(unique_seeds), "test_executions": sum(x["test_count"] for x in reports),
            "code": feature_text, "source_path": os.path.relpath(source_path, root).replace("\\", "/")})
        if len(unique_seeds) != 20:
            missing_seed_keys.append({"program": program, "version": ver, "backend": backend,
                                     "unique_seeds": len(unique_seeds)})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "normalized_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    labels = Counter(x["label"] for x in rows)
    manifest = {"status": "task_manifest_prepared",
        "task": "predict whether Qolumbina DOSS detects a seeded buggy variant on a backend in a majority of available random-seed runs",
        "positive_label": "detected in at least half of report runs; each report is positive if any of its named tests has test_result=fail",
        "negative_label": "not detected in a majority of report runs",
        "source_artifact": str(DATA.relative_to(ROOT)), "published_reports_read": raw_report_count,
        "bugless_control_reports_excluded": ignored_bugless,
        "raw_test_result_counts": dict(status_counts), "observations_after_variant_backend_aggregation": len(rows),
        "program_groups": len(programs), "group_split_unit": "original Qolumbina program; all variants and backends of a program remain together",
        "class_counts": {str(k): v for k, v in labels.items()},
        "variants": len({(x["program"], x["variant_version"]) for x in rows}),
        "backends": sorted({x["backend"] for x in rows}),
        "rows_with_fewer_than_20_seed_reports": missing_seed_keys,
        "feature_policy": "mutant Python source with comments/docstrings removed plus pre-run backend token; no test report output, measured circuit statistic, seed-specific output, or program identity token appended",
        "limitations": ["26 program groups limit the precision of grouped-fold estimates", "label is the majority empirical DOSS detection rate over available seeds, not a ground-truth semantic bug-fix label", "Qolumbina variants are seeded synthetic mutations, not real bug reports"]}
    (OUT / "task_definition.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("task", "published_reports_read", "bugless_control_reports_excluded",
        "raw_test_result_counts", "observations_after_variant_backend_aggregation", "program_groups",
        "class_counts", "variants", "backends", "rows_with_fewer_than_20_seed_reports")}, indent=2))


if __name__ == "__main__":
    main()
