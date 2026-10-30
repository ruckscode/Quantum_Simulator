"""Verify locally acquired quantum datasets without modifying source artifacts.

This creates a compact manifest from archive checks, extraction manifests, git
revisions, and bounded-memory file counts. Hashing multi-GB LintQ data is opt-in.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external_quantum_datasets"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"


def walk(path: Path) -> tuple[int, int, int]:
    files = size = zero = 0
    if not path.exists():
        return files, size, zero
    walk_root = str(path.resolve())
    if os.name == "nt" and not walk_root.startswith("\\\\?\\"):
        walk_root = "\\\\?\\" + walk_root
    for base, dirs, names in os.walk(walk_root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in names:
            p = Path(base) / name
            try:
                n = p.stat().st_size
                files += 1
                size += n
                zero += n == 0
            except OSError:
                pass
    return files, size, zero


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info(path: Path) -> dict:
    if not (path / ".git").exists():
        return {"is_git_checkout": False}
    def run(args: list[str]) -> str:
        return subprocess.run(args, cwd=path, text=True, capture_output=True, check=False).stdout.strip()
    tracked = run(["git", "ls-files", "--cached"])
    lfs = run(["git", "lfs", "ls-files", "-l"]) if shutil.which("git") else ""
    pointers = sum("(LFS: " in line for line in lfs.splitlines())
    return {"is_git_checkout": True, "commit": run(["git", "rev-parse", "HEAD"]),
            "tracked_files": len(tracked.splitlines()), "lfs_files": len(lfs.splitlines()),
            "lfs_pointer_files": pointers}


def extraction(path: Path) -> dict | None:
    mf = path / "_extraction_manifest.json"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text(encoding="utf-8"))
    files, size, zero = walk(path)
    return {"manifest_files_extracted": m.get("files_extracted"),
            "manifest_uncompressed_bytes": m.get("uncompressed_bytes"),
            "actual_files_including_manifest": files,
            "actual_bytes_including_manifest": size,
            "actual_zero_byte_files": zero,
            "manifest_mapping_count": len(m.get("path_mappings", [])),
            "manifest_counts_agree_with_tree": files == int(m.get("files_extracted", -999)) + 1
            and size == int(m.get("uncompressed_bytes", -999)) + mf.stat().st_size}


def zip_extraction_check(archive: Path, destination: Path) -> dict:
    if not archive.exists() or not destination.exists():
        return {"archive_exists": archive.exists(), "destination_exists": destination.exists(), "complete": False}
    manifest_path = destination / "_extraction_manifest.json"
    mapping = {}
    if manifest_path.exists():
        mapping = {x["archive_path"]: x["extracted_path"]
                   for x in json.loads(manifest_path.read_text(encoding="utf-8")).get("path_mappings", [])}
    prefix = str(destination.resolve())
    if os.name == "nt" and not prefix.startswith("\\\\?\\"):
        prefix = "\\\\?\\" + prefix
    missing = []
    with zipfile.ZipFile(archive) as bundle:
        members = [x for x in bundle.infolist() if not x.is_dir()]
        for info in members:
            rel = mapping.get(info.filename, info.filename).replace("/", os.sep)
            target = os.path.join(prefix, rel)
            if not os.path.exists(target):
                missing.append(info.filename)
    return {"archive_regular_files": len(members), "archive_member_bytes": sum(x.file_size for x in members),
            "mapped_member_count": len(mapping), "missing_extracted_members": len(missing),
            "missing_examples": missing[:10], "complete": not missing}


def zip_check(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        members = [i for i in z.infolist() if not i.is_dir()]
        return {"exists": True, "bytes": path.stat().st_size, "members": len(members),
                "member_bytes": sum(i.file_size for i in members),
                "crc_check": "PASS" if bad is None else f"FAIL:{bad}"}


def main() -> None:
    rows = []
    sources = [
        ("Bugs4Q official download cache", ROOT / "data/Bugs4Q_official", "tree"),
        ("Bugs4Q official GitHub archive extraction", ROOT / "data/Bugs4Q_official/extracted/GitHub-Framework", "tree"),
        ("Bugs4Q official Zenodo database extraction", ROOT / "data/Bugs4Q_official/extracted/Zenodo-Database", "tree"),
        ("Bugs4Q official Zenodo database verified extraction", ROOT / "data/Bugs4Q_official/extracted/Zenodo-Database-safe", "tree"),
        ("Bugs4Q official Zenodo framework extraction", ROOT / "data/Bugs4Q_official/extracted/Zenodo-Framework", "tree"),
        ("Bugs4Q framework", BASE / "bugs4q_full/official_framework", "git"),
        ("Bugs4Q database", BASE / "bugs4q_full/official_database", "git"),
        ("Bugs4Q-NA", BASE / "bugs4q_na", "git"),
        ("Bug-Pattern", BASE / "bug_pattern", "git"),
        ("QChecker", BASE / "qchecker", "git"),
        ("Bugs4Q-Robust", ROOT / "data/Bugs4Q_Robust_replication", "tree"),
        ("QMutBench", ROOT / "data/qmutbench_full/data", "tree"),
        ("QSimBench", BASE / "qsimbench/dataset", "git"),
        ("LintQ", BASE / "lintq/extracted", "tree"),
        ("Qolumbina empirical study", BASE / "qolumbina/verified_extracted", "tree"),
        ("Quantum platform bug corpus", BASE / "quantum_platform_bugs_223/case_safe_extracted", "tree"),
        ("QML platform issues", BASE / "qml_platform_bugs/extracted_source", "tree"),
        ("QBugs HornBro subset", BASE / "qbugs_hornbro/hornbro/datasets/Qbugs", "tree"),
        ("Mute-Alg", BASE / "hornbro_mute_alg", "tree"),
    ]
    for name, path, kind in sources:
        count, size, zero = walk(path)
        row = {"dataset": name, "path": str(path.relative_to(ROOT)),
               "exists": path.exists(), "files_excluding_git": count, "bytes": size,
               "zero_byte_files": zero, "status": "DATA_PRESENT" if count else "MISSING"}
        if name == "LintQ":
            row["git_source"] = git_info(BASE / "lintq")
        elif kind == "git":
            row["git"] = git_info(path)
            if row["git"].get("is_git_checkout"):
                row["status"] = "DOWNLOAD_COMPLETE; EXTRACTION_COMPLETE; DATA_PRESENT"
        row["extraction_manifest"] = extraction(path)
        git_verified = bool(row.get("git", {}).get("is_git_checkout"))
        manifest = row["extraction_manifest"]
        extracted_verified = bool(manifest and manifest.get("manifest_counts_agree_with_tree"))
        row["download_status"] = "COMPLETE" if git_verified or path.exists() and count else "BLOCKED"
        row["extraction_status"] = ("NOT_APPLICABLE" if git_verified else
                                    "COMPLETE" if extracted_verified else
                                    "PARTIAL" if path.exists() and count else "BLOCKED")
        row["data_status"] = "PRESENT" if count else "MISSING"
        rows.append(row)

    archives = {
        "Bugs4Q official database ZIP": ROOT / "data/Bugs4Q_official/Bugs4Q-Database.zip",
        "Bugs4Q official GitHub framework ZIP": ROOT / "data/Bugs4Q_official/Bugs4Q-Framework-GitHub-main.zip",
        "Bugs4Q official Zenodo framework ZIP": ROOT / "data/Bugs4Q_official/Bugs4Q-Framework.zip",
        "Qolumbina empirical ZIP": BASE / "qolumbina/qolumbina-empirical-study-main.zip",
        "Quantum platform source ZIP": BASE / "quantum_platform_bugs_223_source.zip",
        "QML platform source ZIP": BASE / "qml_platform_bugs_source.zip",
        "QML platform classified ZIP": BASE / "qml_platform_bugs_classified.zip",
        "QML platform original ZIP": BASE / "qml_platform_bugs_original.zip",
        "QML platform duplicate original-issues ZIP": BASE / "qml_platform_bugs_original_issues.zip",
        "LintQ Figshare TAR": BASE / "lintq/raw/artifact_lintq_dataset.tar",
    }
    archive_results = {}
    for name, path in archives.items():
        check = zip_check(path) if path.suffix.lower() == ".zip" else {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
        if path.exists():
            check["sha256"] = sha256(path)
        archive_results[name] = check
    archive_results["Qolumbina verified extraction"] = zip_extraction_check(
        archives["Qolumbina empirical ZIP"], BASE / "qolumbina/verified_extracted")
    archive_results["Quantum platform verified case-safe extraction"] = zip_extraction_check(
        archives["Quantum platform source ZIP"], BASE / "quantum_platform_bugs_223/case_safe_extracted")
    archive_results["QML platform extraction"] = zip_extraction_check(
        archives["QML platform source ZIP"], BASE / "qml_platform_bugs/extracted_source")
    archive_results["QML classified extraction"] = zip_extraction_check(
        archives["QML platform classified ZIP"], BASE / "qml_platform_bugs/extracted_classified_raw")
    archive_results["QML original extraction"] = zip_extraction_check(
        archives["QML platform original ZIP"], BASE / "qml_platform_bugs/extracted_original_raw")
    archive_results["Bugs4Q official GitHub framework extraction"] = zip_extraction_check(
        archives["Bugs4Q official GitHub framework ZIP"], ROOT / "data/Bugs4Q_official/extracted/GitHub-Framework")
    archive_results["Bugs4Q official Zenodo database extraction"] = zip_extraction_check(
        archives["Bugs4Q official database ZIP"], ROOT / "data/Bugs4Q_official/extracted/Zenodo-Database")
    archive_results["Bugs4Q official Zenodo database verified extraction"] = zip_extraction_check(
        archives["Bugs4Q official database ZIP"], ROOT / "data/Bugs4Q_official/extracted/Zenodo-Database-safe")
    archive_results["Bugs4Q official Zenodo framework extraction"] = zip_extraction_check(
        archives["Bugs4Q official Zenodo framework ZIP"], ROOT / "data/Bugs4Q_official/extracted/Zenodo-Framework")

    unrar = Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe")
    rar_results = {}
    for filename in ("hornbro_qbugs_source.rar", "qbugs_hornbro.rar"):
        path = BASE / filename
        if not path.exists():
            rar_results[filename] = {"exists": False}
            continue
        with path.open("rb") as stream:
            digest = hashlib.md5()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        test = subprocess.run([str(unrar), "t", str(path)], capture_output=True, text=True) if unrar.exists() else None
        rar_results[filename] = {"exists": True, "bytes": path.stat().st_size, "md5": digest.hexdigest(),
                                 "archive_test": "PASS" if test and test.returncode == 0 else
                                 (f"FAIL_EXIT_{test.returncode}" if test else "UNRAR_UNAVAILABLE")}

    # Tighten the summary status fields: a populated folder alone is not proof
    # that its download or extraction completed.
    verified_downloads = {
        "Bugs4Q official download cache": all(archive_results[k].get("crc_check") == "PASS" for k in (
            "Bugs4Q official database ZIP", "Bugs4Q official GitHub framework ZIP", "Bugs4Q official Zenodo framework ZIP")),
        "Bugs4Q official GitHub archive extraction": archive_results["Bugs4Q official GitHub framework ZIP"].get("crc_check") == "PASS",
        "Bugs4Q official Zenodo database extraction": archive_results["Bugs4Q official database ZIP"].get("crc_check") == "PASS",
        "Bugs4Q official Zenodo database verified extraction": archive_results["Bugs4Q official database ZIP"].get("crc_check") == "PASS",
        "Bugs4Q official Zenodo framework extraction": archive_results["Bugs4Q official Zenodo framework ZIP"].get("crc_check") == "PASS",
        "Bugs4Q-Robust": False,
        "QMutBench": False,
        "LintQ": bool(archive_results.get("LintQ Figshare TAR", {}).get("sha256")),
        "Qolumbina empirical study": archive_results["Qolumbina empirical ZIP"].get("crc_check") == "PASS",
        "Quantum platform bug corpus": archive_results["Quantum platform source ZIP"].get("crc_check") == "PASS",
        "QML platform issues": archive_results["QML platform source ZIP"].get("crc_check") == "PASS",
        "QBugs HornBro subset": rar_results.get("hornbro_qbugs_source.rar", {}).get("archive_test") == "PASS",
        "Mute-Alg": False,
    }
    verified_extractions = {
        "Bugs4Q official GitHub archive extraction": archive_results["Bugs4Q official GitHub framework extraction"].get("complete"),
        "Bugs4Q official Zenodo database extraction": archive_results["Bugs4Q official Zenodo database extraction"].get("complete"),
        "Bugs4Q official Zenodo database verified extraction": archive_results["Bugs4Q official Zenodo database verified extraction"].get("complete"),
        "Bugs4Q official Zenodo framework extraction": archive_results["Bugs4Q official Zenodo framework extraction"].get("complete"),
        "Qolumbina empirical study": archive_results["Qolumbina verified extraction"].get("complete"),
        "Quantum platform bug corpus": archive_results["Quantum platform verified case-safe extraction"].get("complete"),
        "QML platform issues": archive_results["QML platform extraction"].get("complete"),
    }
    for row in rows:
        if row.get("git", {}).get("is_git_checkout"):
            row["download_status"] = "COMPLETE"
            row["extraction_status"] = "NOT_APPLICABLE"
        else:
            name = row["dataset"]
            row["download_status"] = "COMPLETE" if verified_downloads.get(name, False) else "PARTIAL"
            row["extraction_status"] = ("COMPLETE" if verified_extractions.get(name, False) else
                                         "NOT_APPLICABLE" if name in {"Bugs4Q official download cache", "Bugs4Q-Robust", "QMutBench", "LintQ", "QBugs HornBro subset", "Mute-Alg"} else "PARTIAL")
        row["overall_status"] = "COMPLETE" if row["download_status"] == "COMPLETE" and row["extraction_status"] in {"COMPLETE", "NOT_APPLICABLE"} and row["data_status"] == "PRESENT" else "PARTIAL"

    robust_csv = ROOT / "data/Bugs4Q_Robust_replication/bugs4q-robust_snapshot/execution_results.csv"
    with robust_csv.open(encoding="utf-8-sig", newline="") as f:
        rr = list(csv.DictReader(f))
    robust_summary = {"execution_rows": len(rr), "issues": len({x.get("id") for x in rr}),
                      "runs": len({x.get("run") for x in rr}),
                      "buggy_outcomes": sorted({x.get("test_buggy") for x in rr}),
                      "fixed_outcomes": sorted({x.get("test_fixed") for x in rr})}

    # QSimBench trace count and line totals, using a streaming reader.
    qsim_root = BASE / "qsimbench/dataset"
    traces = list(qsim_root.rglob("*.jsonl"))
    qsim_lines = 0
    qsim_bytes = 0
    for p in traces:
        qsim_bytes += p.stat().st_size
        with p.open("rb") as f:
            for line in f:
                qsim_lines += bool(line.strip())

    # Dataset status is deliberately distinct from task suitability.
    result = {"verified_at_local": __import__("datetime").datetime.now().astimezone().isoformat(),
              "datasets": rows, "archives": archive_results, "rar_archives": rar_results,
              "Bugs4Q_Robust_execution_csv": robust_summary,
              "QSimBench_jsonl": {"trace_files": len(traces), "nonempty_records": qsim_lines, "trace_bytes": qsim_bytes}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verified_dataset_inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_count": len(rows), "archives_checked": len(archive_results),
                      "Bugs4Q_Robust": robust_summary,
                      "QSimBench": result["QSimBench_jsonl"],
                      "statuses": [{"dataset": x["dataset"], "files": x["files_excluding_git"], "status": x["status"]} for x in rows]}, indent=2))


if __name__ == "__main__":
    main()
