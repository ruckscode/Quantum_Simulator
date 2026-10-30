"""Read-only semantic audit of the local Bug-Pattern and QChecker examples.

This intentionally does not execute source notebooks or Python examples. It
checks whether the checked-in examples support a defensible supervised task.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external_quantum_datasets"
OUT = ROOT / "reports/experiments/qml_source_preprocessing_20261029"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    qroot = BASE / "qchecker"
    qrows = []
    for path in sorted((qroot / "instances").rglob("*.py")):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        try:
            ast.parse(text, filename=path.as_posix())
            syntax = "PARSE_OK"
        except SyntaxError as exc:
            syntax = f"SYNTAX_ERROR:{exc.lineno}:{exc.msg}"
        qrows.append({
            "path": path.relative_to(qroot).as_posix(),
            "directory_label": path.parent.name,
            "sha256": sha(raw),
            "bytes": len(raw),
            "non_whitespace_chars": len(text.strip()),
            "syntax": syntax,
            "validation_status": ("REJECTED_SYNTAX_ERROR" if syntax != "PARSE_OK"
                                  else "UNRESOLVED_NO_VERIFIED_DETECTOR_OUTCOME"),
            "validation_reason": ("The instance does not parse as Python."
                                  if syntax != "PARSE_OK" else
                                  "Folder taxonomy alone does not establish an observed detector outcome or a clean negative; group provenance is not sufficient for leakage-safe evaluation."),
        })

    broot = BASE / "bug_pattern"
    brows = []
    for path in sorted(broot.rglob("*.ipynb")):
        raw = path.read_bytes()
        notebook = json.loads(raw.decode("utf-8"))
        cells = notebook.get("cells", [])
        code = "\n".join("".join(c.get("source", [])) for c in cells
                          if c.get("cell_type") == "code")
        markdown = "\n".join("".join(c.get("source", [])) for c in cells
                              if c.get("cell_type") == "markdown")
        brows.append({
            "path": path.relative_to(broot).as_posix(),
            "top_directory_label": path.relative_to(broot).parts[0],
            "sha256": sha(raw),
            "code_sha256": sha(code.encode("utf-8")),
            "cell_count": len(cells),
            "code_cell_count": sum(c.get("cell_type") == "code" for c in cells),
            "code_chars": len(code),
            "markdown_chars": len(markdown),
            "code_cells_with_outputs": sum(bool(c.get("outputs")) for c in cells
                                            if c.get("cell_type") == "code"),
            "validation_status": "UNRESOLVED_LABEL_LEAKAGE_AND_GROUP_COVERAGE",
            "validation_reason": "Notebook path/title/markdown expose pattern context; no clean negatives are supplied, and this category has insufficient independent examples for a leakage-safe held-out split.",
        })

    qcounts = Counter(row["directory_label"] for row in qrows)
    bcounts = Counter(row["top_directory_label"] for row in brows)
    qbad = [row for row in qrows if row["syntax"] != "PARSE_OK"]
    report = {
        "schema": "bug-pattern-qchecker-source-audit-v1",
        "source_hashes": {
            "qchecker_readme": sha((qroot / "README.md").read_bytes()),
            "qchecker_rules": sha((qroot / "qchecker.py").read_bytes()),
            "qchecker_license": sha((qroot / "LICENSE").read_bytes()),
            "bug_pattern_readme": sha((broot / "README.md").read_bytes()),
            "bug_pattern_license": sha((broot / "LICENSE").read_bytes()),
        },
        "qchecker": {
            "instance_count": len(qrows),
            "directory_label_counts": dict(sorted(qcounts.items())),
            "unique_source_hashes": len({row["sha256"] for row in qrows}),
            "syntax_valid_count": len(qrows) - len(qbad),
            "syntax_errors": qbad,
            "rows": qrows,
            "label_evidence": "The directory taxonomy supplies folder labels; QChecker README enumerates rule codes, but does not certify each file's actual detector outcome or a clean negative label.",
            "eligibility": "INELIGIBLE_FOR_CURRENT_SUPERVISED_TASK",
            "reason": "No clean negatives or verified per-example detector outcomes are included; one example is syntactically invalid; category counts are imbalanced and there are only 31 examples across 9 directories. A random row split would not establish source-group generalization, and grouping by directory would strand rare labels.",
        },
        "bug_pattern": {
            "notebook_count": len(brows),
            "top_directory_counts": dict(sorted(bcounts.items())),
            "unique_notebook_hashes": len({row["sha256"] for row in brows}),
            "unique_code_hashes": len({row["code_sha256"] for row in brows}),
            "notebooks_with_outputs": sum(row["code_cells_with_outputs"] > 0 for row in brows),
            "rows": brows,
            "label_evidence": "README taxonomy and notebook titles provide bug-pattern context, but labels and explanatory text can occur in notebook markdown/title and no clean negatives are supplied.",
            "eligibility": "INELIGIBLE_FOR_CURRENT_SUPERVISED_TASK",
            "reason": "Only 9 local notebooks across 4 categories are present; most external issue/StackOverflow examples linked in the README are not local training rows. Titles/markdown expose the category, there are no clean negatives, and the sample is too small and sparse for a leakage-safe grouped split.",
        },
        "frozen_test_accessed": False,
        "source_execution_performed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "bug_pattern_qchecker_source_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": target.relative_to(ROOT).as_posix(),
        "qchecker_instances": len(qrows),
        "qchecker_syntax_errors": len(qbad),
        "qchecker_labels": dict(sorted(qcounts.items())),
        "bug_pattern_notebooks": len(brows),
        "bug_pattern_categories": dict(sorted(bcounts.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
