import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from experiments import qmutbench_learning_curve as learning
from experiments.qmutbench_outcome_ml import prepare_xy


def _frame(groups=15):
    rows = []
    for group in range(groups):
        for i, label in enumerate(("Survivor", "Survivor", "Survivor", "WOO", "WOO", "OPO")):
            rows.append({
                "Unnamed: 0": len(rows), "Origin_program": f"origin_{group}",
                "algorithm": f"algo_{group % 3}", "qubits": 3 + group % 5,
                "gates": 10 + group * 3, "operation": ("Add", "Replace", "Remove")[i % 3],
                "Position_int": i + 1, "Killed": label, "gate": f"gate_{i % 4}",
                "algorithm_group": f"family_{group % 3}", "dominant_state": f"state_{group % 2}",
                "position": "0-10%",
            })
    return pd.DataFrame(rows)


def test_feature_matrix_excludes_target_record_id_and_origin():
    x, y, groups = prepare_xy(_frame())
    assert "Killed" not in x and "Unnamed: 0" not in x and "Origin_program" not in x
    assert len(x) == len(y) == len(groups)


def test_model_candidates_are_wired_and_gradient_pipeline_handles_unknowns():
    assert isinstance(learning._make_model("random_forest_original", 1).named_steps["model"], RandomForestClassifier)
    assert isinstance(learning._make_model("extra_trees", 1).named_steps["model"], ExtraTreesClassifier)
    hist = learning._make_model("hist_gradient", 1)
    assert isinstance(hist.named_steps["model"], HistGradientBoostingClassifier)
    categorical = hist.named_steps["preprocess"].transformers[1][1]
    assert categorical.named_steps["onehot"].handle_unknown == "ignore"


def test_fixed_folds_are_deterministic_and_representative_prefixes_nested():
    frame = _frame()
    x, y, groups = prepare_xy(frame)
    first = learning._folds_for_dev(x, y, groups, 19)
    second = learning._folds_for_dev(x, y, groups, 19)
    assert all(np.array_equal(a, c) and np.array_equal(b, d)
               for (a, b), (c, d) in zip(first, second))
    order = learning._group_order(frame, 19)
    counts = frame.groupby("Origin_program").size().to_dict()
    prefixes = learning._prefix_groups(order, [30, 60, 90], counts)
    assert set(prefixes["representative_30"]) <= set(prefixes["representative_60"])
    assert set(prefixes["representative_60"]) <= set(prefixes["representative_90"])


def test_fast_group_bootstrap_is_deterministic_and_matches_metric_center():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    pred = np.array([0, 1, 1, 0, 0, 1, 1, 1], dtype=np.int8)
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    result = learning._fast_group_bootstrap(y, pred, groups, seed=4, repetitions=500)
    repeated = learning._fast_group_bootstrap(y, pred, groups, seed=4, repetitions=500)
    assert result == repeated
    assert result["accuracy"][0] <= learning.metrics(y, pred)["accuracy"] <= result["accuracy"][1]


def test_resumes_interrupted_fold_recovers_orphan_and_repairs_corruption(tmp_path, monkeypatch):
    frame = _frame()
    source = tmp_path / "source.csv"
    source.write_text("fixture source identity", encoding="utf-8")
    source_hash = learning.sha256_file(source)
    monkeypatch.setattr(learning, "EXPECTED_SHA256", source_hash)
    monkeypatch.setattr(learning, "load_dataset", lambda _path, **kwargs: frame.copy())
    test_groups = {"origin_14"}
    tr = np.flatnonzero(~frame.Origin_program.isin(test_groups).to_numpy())
    te = np.flatnonzero(frame.Origin_program.isin(test_groups).to_numpy())
    monkeypatch.setattr(learning, "group_holdout", lambda y, groups, **kwargs: (tr, te))
    v3 = tmp_path / "v3"
    v3.mkdir()
    (v3 / "manifest.json").write_text(json.dumps({
        "source_sha256": source_hash,
        "selected_groups": [f"origin_{i}" for i in range(10)],
    }), encoding="utf-8")
    out = tmp_path / "experiment"
    atomic_json = learning._atomic_json
    interrupted = {"done": False}

    def interrupt_after_fit_checkpoint(path, value):
        atomic_json(path, value)
        if path.parent.name == "fold_results" and not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt("simulated interruption after result checkpoint")

    monkeypatch.setattr(learning, "_atomic_json", interrupt_after_fit_checkpoint)
    try:
        learning.run(source, out, seed=19, selected_v3_dir=v3, max_regime="selected_v3")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interruption was not simulated")
    result_files = list((out / "fold_results").glob("selected_v3_fold_1_*.json"))
    assert result_files, "completed fold result should be durable before manifest interruption"
    orphan = result_files[0]

    monkeypatch.setattr(learning, "_atomic_json", atomic_json)
    fit_one = learning._fit_one
    calls = []

    def count_fit(candidate, *args, **kwargs):
        calls.append(candidate)
        return fit_one(candidate, *args, **kwargs)

    monkeypatch.setattr(learning, "_fit_one", count_fit)
    summary = learning.run(source, out, seed=19, selected_v3_dir=v3, max_regime="selected_v3")
    assert summary["regimes"]["selected_v3"]["random_forest_original"]["mean_accuracy"] > 0
    assert len(calls) == 8
    manifest = json.loads((out / "manifest.json").read_text())
    assert any(stage.get("recovered_orphan_checkpoint") for stage in manifest["stages"].values())
    assert len(list((out / "fold_results").glob("selected_v3_fold_*_*.json"))) == 9
    assert all(not (set(stage["train_groups"]) & set(stage["validation_groups"]))
               for name, stage in manifest["stages"].items() if name.startswith("membership:selected_v3:"))
    for fold in range(1, 4):
        memberships = [stage for name, stage in manifest["stages"].items()
                       if name.startswith("membership:") and name.endswith(f"fold_{fold}")]
        assert len({tuple(stage["validation_groups"]) for stage in memberships}) == 1
        assert all("origin_14" not in stage["train_groups"] + stage["validation_groups"]
                   for stage in memberships)

    corrupt = out / "fold_results" / "selected_v3_fold_1_random_forest_original.json"
    corrupt.write_text("corrupt", encoding="utf-8")
    before = len(calls)
    repaired_summary = learning.run(source, out, seed=19, selected_v3_dir=v3, max_regime="selected_v3")
    assert len(calls) == before + 1
    assert json.loads(corrupt.read_text())["status"] == "complete"
    original = summary["regimes"]["selected_v3"]["random_forest_original"]
    repaired = repaired_summary["regimes"]["selected_v3"]["random_forest_original"]
    assert [original[k] for k in ("mean_accuracy", "mean_balanced_accuracy", "mean_macro_f1")] == [
        repaired[k] for k in ("mean_accuracy", "mean_balanced_accuracy", "mean_macro_f1")]
