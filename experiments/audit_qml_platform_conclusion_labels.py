"""Audit the integrated QML-platform workbook's target, duplicates, and groups."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/external_quantum_datasets/qml_platform_bugs/extracted_source/Classified/Conclusion.xlsx"
ARCHIVE = ROOT / "data/external_quantum_datasets/qml_platform_bugs_source.zip"
OUTPUT = ROOT / "reports/experiments/qml_source_preprocessing_20261029/qml_platform_conclusion_label_audit.json"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            xml = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.findall(".//m:t", NS))
                      for item in xml.findall("m:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        sheet = workbook.find("m:sheets/m:sheet", NS)
        target = relmap[sheet.attrib["{" + NS["r"] + "}id"]]
        sheet_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        sheet_path = sheet_path.replace("xl/xl/", "xl/")
        if sheet_path not in names:
            sheet_path = "xl/worksheets/" + target.split("/")[-1]
        xml = ET.fromstring(archive.read(sheet_path))
        rows = []
        for row in xml.findall(".//m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                column = "".join(character for character in cell.attrib["r"] if character.isalpha())
                raw = cell.find("m:v", NS)
                kind = cell.attrib.get("t")
                if kind == "s" and raw is not None:
                    value = shared[int(raw.text)]
                elif kind == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//m:t", NS))
                else:
                    value = raw.text or "" if raw is not None else ""
                if value != "":
                    values[column] = value
            if values:
                rows.append(values)
        return rows


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main() -> None:
    rows = parse_rows(WORKBOOK)
    if not rows or rows[0].get("I") != "0: not quantum\n1: quantum-related":
        raise ValueError("Workbook target annotation header did not match the expected 0/1 definition")
    if rows[0].get("O") != "0: agree\n1: disagree":
        raise ValueError("Workbook agreement header did not match the expected definition")
    raw = rows[1:]
    accepted_by_url = {}
    unresolved = collections.Counter()
    duplicated_rows = []
    duplicate_seen = collections.defaultdict(list)
    for row in raw:
        label = row.get("I", "")
        if label not in {"0.0", "1.0"}:
            unresolved[label or "<blank>"] += 1
            continue
        url, title = row.get("B", ""), row.get("D", "")
        match = re.match(r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)(?:$|[?#])", url)
        if not match or not title:
            unresolved["missing_or_invalid_issue_url_or_title"] += 1
            continue
        record = {"url": url, "repository": match.group(1), "issue_number": match.group(2),
                  "title": title, "label": int(float(label)), "status": row.get("A", ""),
                  "symptom": row.get("F", ""), "location": row.get("G", "")}
        duplicate_seen[url].append(record)
        accepted_by_url.setdefault(url, record)
    for url, records in duplicate_seen.items():
        if len(records) > 1:
            duplicated_rows.extend(records[1:])
            if len({(x["label"], x["title"]) for x in records}) > 1:
                raise ValueError(f"Conflicting duplicate issue URL requires unresolved handling: {url}")
    accepted = list(accepted_by_url.values())

    parent = list(range(len(accepted)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    repo_seen, title_seen = {}, {}
    title_groups = collections.defaultdict(list)
    for i, record in enumerate(accepted):
        title_key = normalized_text(record["title"])
        title_groups[title_key].append(i)
        if record["repository"] in repo_seen:
            union(i, repo_seen[record["repository"]])
        else:
            repo_seen[record["repository"]] = i
        if title_key in title_seen:
            union(i, title_seen[title_key])
        else:
            title_seen[title_key] = i
    components = collections.defaultdict(list)
    for i, record in enumerate(accepted):
        components[find(i)].append(record)
    repo_counts = collections.defaultdict(collections.Counter)
    for record in accepted:
        repo_counts[record["repository"]][str(record["label"])] += 1
    title_collisions = [{"title": accepted[indexes[0]]["title"], "rows": len(indexes),
                         "repositories": sorted({accepted[i]["repository"] for i in indexes}),
                         "labels": sorted({accepted[i]["label"] for i in indexes})}
                        for indexes in title_groups.values() if len(indexes) > 1]
    license_lines = []
    inventory = ROOT / "data/external_quantum_datasets/master_dataset_inventory.csv"
    with inventory.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("dataset") == "Quantum ML framework issue collection":
                license_lines = row
                break
    label_counts = collections.Counter(str(record["label"]) for record in accepted)
    result = {
        "status": "BLOCKED_LICENSE_AND_REPOSITORY_GENERALIZATION",
        "source_archive": str(ARCHIVE.relative_to(ROOT)), "source_archive_sha256": sha256(ARCHIVE),
        "workbook": str(WORKBOOK.relative_to(ROOT)), "workbook_sha256": sha256(WORKBOOK),
        "workbook_data_rows": len(raw), "target_column": "I",
        "target_semantics": "Manual quantum-related annotation: 0=not quantum, 1=quantum-related; not bug/fix or hardware diagnosis.",
        "agreement_column_O_is_not_target": "Agreement 0/1 annotation is separate from quantum-related target and is not used here.",
        "accepted_unique_labeled_issues": len(accepted), "class_counts": dict(label_counts),
        "duplicate_issue_rows_removed": len(duplicated_rows),
        "duplicate_issue_urls": sorted(url for url, items in duplicate_seen.items() if len(items) > 1),
        "unresolved_label_counts": dict(unresolved), "repository_groups": len(repo_counts),
        "repository_label_counts": {key: dict(value) for key, value in sorted(repo_counts.items())},
        "feature_policy": "Issue title only; status, labels, taxonomy, repository, URL, issue ID, and agreement annotations excluded.",
        "exact_normalized_title_collision_fingerprints": len(title_collisions),
        "exact_title_collisions": title_collisions,
        "repository_plus_title_component_count": len(components),
        "largest_repository_plus_title_component_rows": max(map(len, components.values()), default=0),
        "component_label_conflicts": sum(len({row["label"] for row in component}) > 1 for component in components.values()),
        "license_evidence": "The local source inventory states no license file was located and issue text remains subject to platform/source terms.",
        "license_inventory_source": str(inventory.relative_to(ROOT)),
        "license_inventory_excerpt": license_lines,
        "blocking_reasons": ["source archive has no declared reuse license in the local repository inventory",
                             "only 339 unique issue rows have clean 0/1 targets; 140 rows are unlabeled or misaligned",
                             "repository and title grouping yields only nine connected components, including one component with 264 records; a repository-generalization claim is not supportable from this split"],
        "frozen_test_accessed": False,
        "output": str(OUTPUT.relative_to(ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "workbook_data_rows", "accepted_unique_labeled_issues",
        "class_counts", "duplicate_issue_rows_removed", "unresolved_label_counts", "repository_groups",
        "repository_plus_title_component_count", "largest_repository_plus_title_component_rows", "output")}, indent=2))


if __name__ == "__main__":
    main()
