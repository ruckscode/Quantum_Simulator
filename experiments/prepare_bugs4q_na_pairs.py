"""Resolve README-documented Bugs4Q-NA buggy/fixed Python pairs."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import tokenize
from io import StringIO
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_quantum_datasets/bugs4q_na"
CORE = ROOT / "data/Bugs4Q_official/extracted/GitHub-Framework/Bugs4Q-Framework-main/qiskit"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_na"


def sanitize(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(raw)
        doc_lines = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_lines.update(range(val.lineno, getattr(val, "end_lineno", val.lineno) + 1))
        toks = tokenize.generate_tokens(StringIO(raw).readline)
        code = " ".join(t.string for t in toks if t.type != tokenize.COMMENT and
                         not (t.type == tokenize.STRING and t.start[0] in doc_lines) and
                         t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                        tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        code = re.sub(r"\s+", " ", raw)
    return raw, code


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Existing core issue identifiers make overlap removal explicit.
    core_urls = set()
    for info in CORE.glob("*/info_*.csv"):
        try:
            frame = pd.read_csv(info, sep=";", dtype=str)
            if "Url" in frame:
                core_urls.update(frame["Url"].dropna().str.lower())
        except Exception:
            pass
    seen_issue, rows, skipped = set(), [], []
    for line_no, line in enumerate((DATA / "README.md").read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "|" not in line or "http" not in line.lower() or "buggy" not in line.lower() or "fixed" not in line.lower():
            continue
        links = re.findall(r"\[[^]]*\]\(([^)]+)\)", line)
        urls = [u.rstrip(".,;") for u in links if u.startswith("http")]
        local = [unquote(u) for u in links if not u.startswith("http") and u != "---"]
        if not urls or len(local) < 2:
            continue
        issue_url = urls[0].lower()
        # Keep only one representation of each documented issue URL.
        if issue_url in seen_issue:
            continue
        seen_issue.add(issue_url)
        if issue_url in core_urls:
            skipped.append({"line": line_no, "issue_url": issue_url, "reason": "overlaps official 31-issue core subset"})
            continue
        pair = []
        for rel in local[:2]:
            p = (DATA / rel.lstrip("./")).resolve()
            if not p.is_relative_to(DATA.resolve()) or not p.is_file():
                pair = []
                break
            pair.append(p)
        if len(pair) != 2 or any(p.suffix.lower() != ".py" for p in pair):
            skipped.append({"line": line_no, "issue_url": issue_url, "reason": "missing or non-Python source pair"})
            continue
        if pair[0].resolve() == pair[1].resolve() or pair[0].read_bytes() == pair[1].read_bytes():
            skipped.append({"line": line_no, "issue_url": issue_url, "reason": "buggy/fixed links resolve to same content"})
            continue
        for path, label in zip(pair, (1, 0)):
            raw, code = sanitize(path)
            try:
                ast.parse(raw)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
            group_id = str(int(hashlib.sha256(issue_url.encode()).hexdigest()[:12], 16))
            rows.append({"record_id": f"b4qna-{group_id}-{'buggy' if label else 'fixed'}",
                         "bug_id": group_id,
                         "issue_url": issue_url, "label": label, "code": code,
                         "source_path": str(path.relative_to(ROOT)),
                         "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                         "syntax_valid": syntax_valid})
    # A source snapshot can recur under several issue links. Exclude only exact duplicate
    # source states after README issue grouping and record every exclusion.
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["label", "raw_sha256"], keep="first")
    counts = df.groupby("bug_id").label.nunique()
    good = counts[counts == 2].index
    dropped_groups = sorted(set(df.bug_id) - set(good))
    df = df[df.bug_id.isin(good)].reset_index(drop=True)
    df.to_csv(OUT / "normalized_manifest.csv", index=False)
    (OUT / "preparation_manifest.json").write_text(json.dumps({
        "status": "prepared_grouped_buggy_fixed_source_task", "dataset": "Bugs4Q-NA",
        "source_readme": str((DATA / "README.md").relative_to(ROOT)),
        "label_map": {"fixed": 0, "buggy": 1}, "group_key": "source issue URL hash",
        "core_overlap_issue_urls_excluded": len([x for x in skipped if "overlaps" in x["reason"]]),
        "rows": len(df), "paired_issue_groups": int(df.bug_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
        "syntax_valid": int(df.syntax_valid.sum()), "exact_duplicate_or_invalid_groups_removed": dropped_groups,
        "skipped": skipped,
        "limitations": "Only README-linked Python pairs are used; Q#, Cirq files in other languages, notebook ambiguity, and undocumented files are excluded. Exact duplicate source states are removed. Pair labels are repository-documented, not independently revalidated.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(df), "groups": int(df.bug_id.nunique()),
                      "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
                      "manifest": str(OUT / "preparation_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
