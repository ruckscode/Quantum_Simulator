"""Grouped buggy/fixed source classification on the official Bugs4Q Qiskit subset."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import tokenize
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/Bugs4Q_official/extracted/GitHub-Framework/Bugs4Q-Framework-main/qiskit"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_standalone"
SEED = 20261028


def sanitized_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
        docstring_spans = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    for line in range(value.lineno, getattr(value, "end_lineno", value.lineno) + 1):
                        docstring_spans.add(line)
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        kept = [t for t in tokens if t.type != tokenize.COMMENT and
                not (t.type == tokenize.STRING and t.start[0] in docstring_spans) and t.type not in (
                 tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                 tokenize.ENCODING, tokenize.ENDMARKER)]
        return " ".join(t.string for t in kept)
    except SyntaxError:
        # Some buggy snapshots are deliberately ill-formed Python. Tokenization
        # still removes comments without requiring the source to parse.
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        kept = [t for t in tokens if t.type not in (
            tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER)]
        return " ".join(t.string for t in kept)
    except (tokenize.TokenError, IndentationError):
        return re.sub(r"\s+", " ", source)


def metrics(y, pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
        confusion_matrix, f1_score, precision_recall_fscore_support)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_labels_fixed0_buggy1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {str(i): {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)}}


def collect():
    rows = []
    for folder in sorted((p for p in DATA.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        bug = folder.name
        bug_file, fixed_file = folder / f"buggy_{bug}.py", folder / f"fixed_{bug}.py"
        test_bug, test_fixed = folder / f"test_b{bug}.py", folder / f"test_f{bug}.py"
        info = folder / f"info_{bug}.csv"
        if not (bug_file.is_file() and fixed_file.is_file() and test_bug.is_file() and test_fixed.is_file() and info.is_file()):
            continue
        for path, label in ((fixed_file, 0), (bug_file, 1)):
            code = sanitized_source(path)
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="replace"))
                syntax_ok = True
            except SyntaxError:
                syntax_ok = False
            rows.append({"record_id": f"qiskit-{bug}-{'fixed' if label == 0 else 'buggy'}",
                         "bug_id": bug, "label": label, "code": code, "source_path": str(path.relative_to(ROOT)),
                         "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "test_path": str((test_fixed if label == 0 else test_bug).relative_to(ROOT)),
                         "test_sha256": hashlib.sha256((test_fixed if label == 0 else test_bug).read_bytes()).hexdigest(),
                         "syntax_valid": syntax_ok})
    return rows


def run():
    import psutil
    start_ram = psutil.virtual_memory().available / 2**20
    if start_ram < 600:
        raise MemoryError(f"Standalone benchmark needs 600 MiB preflight; found {start_ram:.1f} MiB")
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import LinearSVC, SVC
    from sklearn.tree import DecisionTreeClassifier
    available_after_import = psutil.virtual_memory().available / 2**20
    if available_after_import < 500:
        raise MemoryError(f"Stopped after imports below 500 MiB floor: {available_after_import:.1f} MiB")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    frame = pd.DataFrame(rows)
    if len(frame) < 40 or frame.bug_id.nunique() < 20:
        raise ValueError("Too few complete paired/tested Bugs4Q Qiskit cases")
    counts = frame.groupby("bug_id").label.nunique()
    groups = sorted(counts[counts == 2].index, key=int)
    frame = frame[frame.bug_id.isin(groups)].reset_index(drop=True)
    y = frame.label.to_numpy(dtype=int)
    g = frame.bug_id.to_numpy()
    if len(np.unique(y)) != 2 or frame.groupby("bug_id").label.nunique().min() != 2:
        raise ValueError("Every retained bug must have one fixed and one buggy program")
    frame.to_csv(OUT / "normalized_manifest.csv", index=False)
    folds = GroupKFold(n_splits=5)
    split_rows, pred_rows, metrics_rows = [], [], []
    configs = [
        ("majority", DummyClassifier(strategy="prior"), "tfidf"),
        ("logistic_tfidf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED), "tfidf"),
        ("linear_svc_tfidf", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), "tfidf"),
        ("rbf_svc_tfidf", SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=SEED), "tfidf"),
        ("decision_tree_4d", DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=SEED), "4d"),
        ("random_forest_4d", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=SEED), "4d"),
        ("extra_trees_4d", ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=SEED), "4d"),
        ("logistic_4d", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=SEED), "4d"),
        ("hist_gradient_boosting_4d", HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=7, l2_regularization=1.0, random_state=SEED), "4d"),
        ("small_mlp_4d", MLPClassifier(hidden_layer_sizes=(16,), alpha=0.01, max_iter=300, random_state=SEED), "4d"),
    ]
    started = time.perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(folds.split(frame.code, y, g), start=1):
        if psutil.virtual_memory().available / 2**20 < 500:
            raise MemoryError("Available RAM fell below 500 MiB between folds")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=1,
                                     sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        xtr_sparse = vectorizer.fit_transform(frame.code.iloc[train_idx])
        xva_sparse = vectorizer.transform(frame.code.iloc[valid_idx])
        n_comp = min(4, xtr_sparse.shape[0] - 1, xtr_sparse.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=SEED + fold)
        ztr = svd.fit_transform(xtr_sparse)
        zva = svd.transform(xva_sparse)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(ztr)
        xtr_4d, xva_4d = scaler.transform(ztr), scaler.transform(zva)
        ytr, yva = y[train_idx], y[valid_idx]
        if set(ytr) != {0, 1} or set(yva) != {0, 1}:
            raise ValueError(f"fold {fold} lacks a class")
        split_rows.append({"fold": fold, "train_bug_ids": sorted(set(map(str, g[train_idx]))),
                           "validation_bug_ids": sorted(set(map(str, g[valid_idx]))),
                           "train_records": len(train_idx), "validation_records": len(valid_idx),
                           "group_overlap": len(set(g[train_idx]) & set(g[valid_idx]))})
        for name, estimator, schema in configs:
            t0 = time.perf_counter()
            a, b = (xtr_sparse, xva_sparse) if schema == "tfidf" else (xtr_4d, xva_4d)
            estimator.fit(a, ytr)
            pred = estimator.predict(b)
            m = metrics(yva, pred)
            metrics_rows.append({"model": name, "fold": fold, **m,
                                 "runtime_seconds": time.perf_counter() - t0,
                                 "features": "fold-local TF-IDF 1-2 grams" if schema == "tfidf" else "fold-local TF-IDF + TruncatedSVD(4) + MinMax"})
            for j, idx in enumerate(valid_idx):
                pred_rows.append({"record_id": frame.record_id.iloc[idx], "bug_id": g[idx], "fold": fold,
                                  "y_true": int(yva[j]), "model": name, "y_pred": int(pred[j])})
    # Aggregate OOF records and group-bootstrap uncertainty for every model.
    pred_df = pd.DataFrame(pred_rows)
    aggregate = []
    rng = np.random.default_rng(SEED)
    bug_ids = np.asarray(sorted(set(map(str, g)), key=int), dtype=object)
    group_rows = {bid: np.flatnonzero(frame.bug_id.astype(str).to_numpy() == bid) for bid in bug_ids}
    oof_wide = pred_df.pivot(index=["record_id", "bug_id", "fold", "y_true"], columns="model", values="y_pred").reset_index()
    for name, _, schema in configs:
        pred = oof_wide[name].to_numpy(dtype=int)
        yall = oof_wide.y_true.to_numpy(dtype=int)
        row = {"model": name, "feature_schema": schema, **metrics(yall, pred),
               "fold_macro_f1_mean": float(pd.DataFrame(metrics_rows).query("model == @name").macro_f1.mean()),
               "fold_macro_f1_sd": float(pd.DataFrame(metrics_rows).query("model == @name").macro_f1.std(ddof=1))}
        origin_ids = oof_wide.bug_id.astype(str).to_numpy()
        boot = []
        for _ in range(1000):
            draw = rng.choice(bug_ids, size=len(bug_ids), replace=True)
            ix = np.concatenate([np.flatnonzero(origin_ids == bid) for bid in draw])
            boot.append(float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(yall[ix], pred[ix], average="macro", zero_division=0)))
        row["bug_group_bootstrap_95ci_macro_f1"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        aggregate.append(row)
    (OUT / "fold_metrics.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    pd.DataFrame(aggregate).to_json(OUT / "leaderboard.json", orient="records", indent=2)
    pred_df.to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_json(OUT / "grouped_split_manifest.json", orient="records", indent=2)
    origin_metrics = []
    for bid, part in pred_df.groupby("bug_id"):
        for name, grp in part.groupby("model"):
            origin_metrics.append({"bug_id": str(bid), "model": name, "records": len(grp),
                                   **metrics(grp.y_true.to_numpy(), grp.y_pred.to_numpy())})
    pd.DataFrame(origin_metrics).to_json(OUT / "per_bug_metrics.json", orient="records", indent=2)
    # Refit the top grouped-OOF candidate on all eligible examples after CV.
    from sklearn.base import clone
    best_row = max(aggregate, key=lambda row: row["macro_f1"])
    best_name, _, best_schema = next(c for c in configs if c[0] == best_row["model"])
    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=1,
                                       sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    final_x = final_vectorizer.fit_transform(frame.code)
    checkpoint = {"vectorizer": final_vectorizer, "model_name": best_name,
                  "target_map": {"fixed": 0, "buggy": 1}, "training_bug_ids": groups,
                  "feature_policy": "comments and docstrings stripped; executable string literals retained; TF-IDF vocabulary fit on all eligible data after OOF evaluation"}
    if best_schema == "tfidf":
        final_model = clone(next(c[1] for c in configs if c[0] == best_name)).fit(final_x, y)
        checkpoint["model"] = final_model
        checkpoint["feature_schema"] = "TF-IDF 1-2 grams"
    else:
        final_svd = TruncatedSVD(n_components=min(4, final_x.shape[0]-1, final_x.shape[1]-1),
                                 n_iter=7, random_state=SEED)
        final_z = final_svd.fit_transform(final_x)
        final_scaler = MinMaxScaler(feature_range=(0, 1)).fit(final_z)
        final_model = clone(next(c[1] for c in configs if c[0] == best_name)).fit(final_scaler.transform(final_z), y)
        checkpoint.update({"model": final_model, "svd": final_svd, "scaler": final_scaler,
                           "feature_schema": "TF-IDF -> TruncatedSVD(4) -> MinMaxScaler"})
    import joblib
    joblib.dump(checkpoint, OUT / "best_classical_checkpoint.joblib")
    summary = {"status": "completed_classical_grouped_oof", "source": "Bugs4Q Version-1.0.0 Qiskit examples",
        "source_repository": "https://github.com/Z-726/Bugs4Q-Framework", "license": "Apache-2.0 (framework repository)",
        "task": "binary classification of source snapshots: fixed=0, buggy=1; source-native bug/fix state, not QMutBench labels",
        "independent_group": "bug_id; buggy and fixed snapshots from each issue stay in one fold",
        "retained_bug_groups": int(frame.bug_id.nunique()), "records": int(len(frame)),
        "complete_trigger_test_pairs": int(frame.bug_id.nunique()), "excluded_incomplete_pairs": "Cases without both buggy/fixed sources, both corresponding trigger tests, and metadata were excluded.",
        "all_source_parses": bool(frame.syntax_valid.all()), "class_counts": {str(k): int(v) for k,v in frame.label.value_counts().sort_index().items()},
        "feature_leakage_controls": "Comments and docstrings stripped; executable string literals retained because they carry program semantics; ID/path/bug metadata not predictors; TF-IDF vocabulary, SVD, and scaling fit within each training fold.",
        "quantum_model": "Not run in this stage; classical standalone first. Candidate is eligible for bounded QNN follow-up if memory permits.",
        "best_candidate": best_row,
        "best_checkpoint": "best_classical_checkpoint.joblib; top grouped-OOF candidate retrained on all 31 groups after evaluation",
        "runtime_seconds": time.perf_counter()-started, "available_ram_start_mib": start_ram,
        "available_ram_after_import_mib": available_after_import, "available_ram_end_mib": psutil.virtual_memory().available/2**20,
        "results": aggregate}
    (OUT / "standalone_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
