"""Train matched supervised quantum/hybrid models on Bugs4Q bug groups."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_standalone/normalized_manifest.csv"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_standalone/qnn"
SEED = 20261028
STEPS = 15
MIN_PRE_RAM_MIB = 700
RAM_FLOOR_MIB = 500


def _score(y, pred, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix,
           precision_recall_fscore_support):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_labels_fixed0_buggy1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {str(i): {"precision": float(p[i]), "recall": float(r[i]),
                                    "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)}}


def _make_qnn(kind, seed, qml, pnp, np):
    hybrid = kind == "angle_hybrid_readout"
    amplitude = kind == "amplitude_vqc"
    n_qubits = 2 if amplitude else 4
    depth = 2
    device = qml.device("default.qubit", wires=n_qubits, shots=None, seed=seed)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def circuit(x, weights):
        if amplitude:
            qml.AmplitudeEmbedding(x + 1e-7, wires=range(n_qubits), normalize=True)
        for layer in range(depth):
            if not amplitude and (layer == 0 or kind == "angle_reupload_vqc"):
                qml.AngleEmbedding(pnp.pi * x, wires=range(n_qubits), rotation="Y")
            for wire in range(n_qubits):
                qml.Rot(*weights[layer, wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return tuple(qml.expval(qml.PauliZ(w)) for w in range(n_qubits))

    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.normal(0.0, 0.12, (depth, n_qubits, 3)), requires_grad=True)
    initial_weights = np.asarray(weights).copy()
    if hybrid:
        head = pnp.array(rng.normal(0.0, 0.1, n_qubits), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)
        initial_head, initial_bias = np.asarray(head).copy(), float(bias)
    else:
        head = bias = None
        initial_head = initial_bias = None

    def probability(row, w, h=None, b=None):
        z = pnp.stack(circuit(row, w))
        if hybrid:
            logit = pnp.dot(h, z) + b
            return 1.0 / (1.0 + pnp.exp(-logit))
        return pnp.clip((1.0 - z[0]) / 2.0, 1e-7, 1.0 - 1e-7)

    return {"kind": kind, "device": device, "circuit": circuit, "weights": weights,
            "initial_weights": initial_weights, "head": head, "bias": bias,
            "initial_head": initial_head, "initial_bias": initial_bias,
            "probability": probability, "n_qubits": n_qubits, "depth": depth,
            "trainable_parameters": int(weights.size + (n_qubits + 1 if hybrid else 0))}


def _fit_qnn(model, x, y, seed, qml, pnp, np):
    x = pnp.array(x, requires_grad=False)
    y = pnp.array(y, requires_grad=False, dtype=int)
    counts = pnp.array([pnp.sum(y == c) for c in (0, 1)], requires_grad=False)
    class_weights = len(y) / (2.0 * counts)
    row_weights = class_weights[y]
    hybrid = model["kind"] == "angle_hybrid_readout"
    weights, head, bias = model["weights"], model["head"], model["bias"]

    def objective(w, h=None, b=None, xx=None, yy=None, sw=None):
        features = x if xx is None else xx
        labels = y if yy is None else yy
        weights_for_rows = row_weights if sw is None else sw
        probs = pnp.stack([model["probability"](row, w, h, b) for row in features])
        loss = -(labels * pnp.log(probs) + (1.0 - labels) * pnp.log(1.0 - probs))
        return pnp.mean(weights_for_rows * loss)

    # Check an autodiff coordinate against a centered finite difference before fitting.
    x0, y0, sw0 = x[0], float(y[0]), float(row_weights[0])
    if hybrid:
        def one_loss(w, h, b):
            p = model["probability"](x0, w, h, b)
            return -sw0 * (y0 * pnp.log(p) + (1.0-y0) * pnp.log(1.0-p))
        grads = qml.grad(one_loss, argnum=(0, 1, 2))(weights, head, bias)
        g = float(grads[0][0, 0, 0])
        eps = 1e-5
        wp = pnp.array(np.asarray(weights).copy(), requires_grad=True)
        wm = pnp.array(np.asarray(weights).copy(), requires_grad=True)
        wp[0, 0, 0] += eps; wm[0, 0, 0] -= eps
        fd = float((one_loss(wp, head, bias) - one_loss(wm, head, bias)) / (2 * eps))
        grad_norm = float(np.sqrt(sum(np.square(np.asarray(gx)).sum() for gx in grads)))
    else:
        def one_loss(w):
            p = model["probability"](x0, w)
            return -sw0 * (y0 * pnp.log(p) + (1.0-y0) * pnp.log(1.0-p))
        grads = qml.grad(one_loss)(weights)
        g = float(grads[0, 0, 0])
        eps = 1e-5
        wp = pnp.array(np.asarray(weights).copy(), requires_grad=True)
        wm = pnp.array(np.asarray(weights).copy(), requires_grad=True)
        wp[0, 0, 0] += eps; wm[0, 0, 0] -= eps
        fd = float((one_loss(wp) - one_loss(wm)) / (2 * eps))
        grad_norm = float(np.linalg.norm(np.asarray(grads)))
    grad_error = abs(g - fd)
    grad_check = {"autodiff_coordinate": g, "finite_difference_coordinate": fd,
                  "absolute_error": grad_error, "gradient_l2_norm": grad_norm,
                  "passed": bool(np.isfinite(grad_norm) and grad_norm > 0 and grad_error < 2e-3)}
    if not grad_check["passed"]:
        raise ArithmeticError(f"Gradient check failed for {model['kind']}: {grad_check}")

    optimizer = qml.AdamOptimizer(stepsize=0.06)
    loss_history = []
    started = time.perf_counter()
    with qml.Tracker(model["device"]) as tracker:
        for _ in range(STEPS):
            if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                raise MemoryError("Bugs4Q QNN fit stopped at the 500 MiB safety floor")
            if hybrid:
                (weights, head, bias), cost = optimizer.step_and_cost(objective, weights, head, bias)
            else:
                weights, cost = optimizer.step_and_cost(objective, weights)
            loss_history.append(float(cost))
    model.update({"weights": weights, "head": head, "bias": bias,
                  "loss_history": loss_history, "training_seconds": time.perf_counter()-started,
                  "training_executions": int(tracker.totals.get("executions", 0)),
                  "gradient_check": grad_check,
                  "weight_update_l2": float(np.linalg.norm(np.asarray(weights)-model["initial_weights"])),
                  "head_update_l2": float(np.linalg.norm(np.asarray(head)-model["initial_head"])) if hybrid else 0.0})
    return model


def run():
    before = psutil.virtual_memory().available / 2**20
    if before < MIN_PRE_RAM_MIB:
        raise MemoryError(f"Bugs4Q QNN requires {MIN_PRE_RAM_MIB} MiB preflight; found {before:.1f} MiB")
    import numpy as np
    import pandas as pd
    import pennylane as qml
    from pennylane import numpy as pnp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
        f1_score, precision_recall_fscore_support)
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MinMaxScaler
    after_import = psutil.virtual_memory().available / 2**20
    if after_import < MIN_PRE_RAM_MIB:
        raise MemoryError(f"Post-import Bugs4Q QNN headroom below {MIN_PRE_RAM_MIB} MiB: {after_import:.1f} MiB")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite QNN campaign outputs: {OUT}")
    OUT.mkdir(parents=True)
    df = pd.read_csv(IN)
    y = df.label.to_numpy(dtype=int)
    groups = df.bug_id.to_numpy()
    split = GroupKFold(n_splits=5)
    kinds = ["angle_vqc", "angle_reupload_vqc", "angle_hybrid_readout", "amplitude_vqc"]
    predictions, fold_results, history, checkpoints = [], [], [], []
    classical_oof = {name: [] for name in ("logistic_4d", "random_forest_4d", "extra_trees_4d")}
    started_all = time.perf_counter()

    for fold, (train, valid) in enumerate(split.split(df.code, y, groups), 1):
        if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
            raise MemoryError("Available RAM below 500 MiB before a Bugs4Q fold")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=1,
                                     sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        xtr_sparse = vectorizer.fit_transform(df.code.iloc[train])
        xva_sparse = vectorizer.transform(df.code.iloc[valid])
        svd = TruncatedSVD(n_components=4, n_iter=7, random_state=SEED+fold)
        ztr, zva = svd.fit_transform(xtr_sparse), svd.transform(xva_sparse)
        scaler = MinMaxScaler().fit(ztr)
        xtr, xva = scaler.transform(ztr), scaler.transform(zva)
        ytr, yva = y[train], y[valid]
        if set(ytr) != {0, 1} or set(yva) != {0, 1}:
            raise ValueError(f"Bug-grouped fold {fold} lacks a class")
        import joblib
        joblib.dump({"vectorizer": vectorizer, "svd": svd, "scaler": scaler,
                     "feature_names": [f"svd_{i}" for i in range(4)],
                     "training_bug_ids": sorted(set(map(str, groups[train])))},
                    OUT / f"preprocessing_fold{fold}.joblib")

        controls = [("logistic_4d", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=SEED)),
                    ("random_forest_4d", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2,
                        class_weight="balanced", n_jobs=1, random_state=SEED)),
                    ("extra_trees_4d", ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2,
                        class_weight="balanced", n_jobs=1, random_state=SEED))]
        for name, estimator in controls:
            t0 = time.perf_counter(); estimator.fit(xtr, ytr)
            prob = estimator.predict_proba(xva)[:, 1]
            pred = (prob >= 0.5).astype(int)
            classical_oof[name].extend({"record_id":df.record_id.iloc[ix],"bug_id":str(groups[ix]),"fold":fold,
                "y_true":int(y[ix]),"y_pred":int(pred[j]),"prob_fixed":float(1-prob[j]),"prob_buggy":float(prob[j])}
                for j, ix in enumerate(valid))
            fold_results.append({"model":name,"fold":fold,"runtime_seconds":time.perf_counter()-t0,
                "train_bugs":len(set(groups[train])),"validation_bugs":len(set(groups[valid])),
                "group_overlap":len(set(groups[train])&set(groups[valid])),
                **_score(yva,pred,f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support)})

        for ki, kind in enumerate(kinds):
            seed = SEED + fold*100 + ki
            model = _make_qnn(kind, seed, qml, pnp, np)
            model = _fit_qnn(model, xtr, ytr, seed, qml, pnp, np)
            probs = []
            for row in xva:
                z = np.asarray(model["circuit"](pnp.array(row), model["weights"]), dtype=float)
                if kind == "angle_hybrid_readout":
                    logit = float(np.dot(np.asarray(model["head"]), z) + float(model["bias"]))
                    p1 = 1.0/(1.0+np.exp(-logit))
                else:
                    p1 = np.clip((1.0-z[0])/2.0, 0.0, 1.0)
                probs.append([1.0-p1,p1])
            probs = np.asarray(probs); pred = probs.argmax(axis=1)
            common = {"model":kind,"fold":fold,"train_bugs":len(set(groups[train])),
                "validation_bugs":len(set(groups[valid])),"train_records":len(train),"validation_records":len(valid),
                "group_overlap":len(set(groups[train])&set(groups[valid])),"qubits":model["n_qubits"],
                "depth":model["depth"],"trainable_parameters":model["trainable_parameters"],
                "encoding":"amplitude" if kind=="amplitude_vqc" else "angle",
                "data_reuploading":kind=="angle_reupload_vqc","trainable_classical_readout":kind=="angle_hybrid_readout",
                "training_seconds":model["training_seconds"],"training_circuit_executions":model["training_executions"],
                "loss_history":model["loss_history"],"loss_drop":model["loss_history"][0]-model["loss_history"][-1],
                "weight_update_l2":model["weight_update_l2"],"head_update_l2":model["head_update_l2"],
                "gradient_check":model["gradient_check"],
                **_score(yva,pred,f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support)}
            fold_results.append(common); history.append({"model":kind,"fold":fold,"loss":model["loss_history"]})
            predictions.extend({"record_id":df.record_id.iloc[ix],"bug_id":str(groups[ix]),"fold":fold,
                "y_true":int(y[ix]),"model":kind,"y_pred":int(pred[j]),
                "prob_fixed":float(probs[j,0]),"prob_buggy":float(probs[j,1])} for j,ix in enumerate(valid))
            checkpoints.append({"model":kind,"fold":fold,"weights":np.asarray(model["weights"]).tolist(),
                "initial_weights":model["initial_weights"].tolist(),
                "readout_weights":np.asarray(model["head"]).tolist() if kind=="angle_hybrid_readout" else None,
                "readout_bias":float(model["bias"]) if kind=="angle_hybrid_readout" else None,
                "initial_readout_weights":model["initial_head"].tolist() if kind=="angle_hybrid_readout" else None,
                "initial_readout_bias":model["initial_bias"],"seed":seed,"steps":STEPS,
                "learning_rate":0.06,"qubits":model["n_qubits"],"depth":model["depth"],
                "encoding":"amplitude" if kind=="amplitude_vqc" else "angle",
                "data_reuploading":kind=="angle_reupload_vqc","trainable_classical_readout":kind=="angle_hybrid_readout"})
            # Persist after every trained candidate so an interrupted campaign retains real work.
            (OUT/"fold_checkpoints.json").write_text(json.dumps(checkpoints,indent=2),encoding="utf-8")
            (OUT/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            pd.DataFrame(fold_results).to_json(OUT/"fold_metrics.json",orient="records",indent=2)
            pd.DataFrame(predictions).to_csv(OUT/"qnn_oof_predictions_partial.csv",index=False)
            if psutil.virtual_memory().available/2**20 < RAM_FLOOR_MIB:
                raise MemoryError("Available RAM below 500 MiB after a QNN fold")

    # Aggregate metrics and save OOF predictions for each QNN and matched control.
    predictions.extend({**r,"model":name} for name,rows in classical_oof.items() for r in rows)
    pred_frame = pd.DataFrame(predictions)
    pred_frame.to_csv(OUT/"oof_predictions.csv",index=False)
    pd.DataFrame(fold_results).to_json(OUT/"fold_metrics.json",orient="records",indent=2)
    (OUT/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (OUT/"fold_checkpoints.json").write_text(json.dumps(checkpoints,indent=2),encoding="utf-8")
    models = list(classical_oof) + kinds
    aggregate=[]; per_class={}; rng=np.random.default_rng(SEED)
    bug_ids=np.array(sorted(set(map(str,groups)),key=int),dtype=object)
    for name in models:
        subset=pred_frame[pred_frame.model==name].sort_values(["record_id"])
        yy=subset.y_true.to_numpy(); pp=subset.y_pred.to_numpy()
        row={"model":name,"family":"classical" if name in classical_oof else "quantum_or_hybrid",
             **_score(yy,pp,f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support)}
        intervals=[]
        for _ in range(2000):
            draw=rng.choice(bug_ids,size=len(bug_ids),replace=True)
            parts=[subset[subset.bug_id==bid] for bid in draw]
            boot=pd.concat(parts,ignore_index=True)
            intervals.append(float(f1_score(boot.y_true,boot.y_pred,average="macro",zero_division=0)))
        row["bug_group_bootstrap_95ci_macro_f1"]=[float(np.quantile(intervals,.025)),float(np.quantile(intervals,.975))]
        aggregate.append(row); per_class[name]=row["per_class"]
    pd.DataFrame(aggregate).to_json(OUT/"leaderboard.json",orient="records",indent=2)
    (OUT/"per_class_metrics.json").write_text(json.dumps(per_class,indent=2),encoding="utf-8")
    per_bug=[]
    for (bid,name),part in pred_frame.groupby(["bug_id","model"]):
        per_bug.append({"bug_id":str(bid),"model":name,"records":len(part),
            **_score(part.y_true.to_numpy(),part.y_pred.to_numpy(),f1_score,accuracy_score,balanced_accuracy_score,
                     confusion_matrix,precision_recall_fscore_support)})
    pd.DataFrame(per_bug).to_json(OUT/"per_bug_metrics.json",orient="records",indent=2)
    summary={"status":"completed_grouped_supervised_qnn_benchmark","task":"Bugs4Q source state: fixed=0, buggy=1",
        "data":"31 issue groups / 62 paired snapshots; comments and docstrings stripped; executable strings retained",
        "group_split":"5-fold GroupKFold by bug_id; buggy/fixed pair stays within the same fold",
        "qnn_candidates":kinds,"matched_classical_controls":list(classical_oof),"steps_per_fold":STEPS,
        "optimizer":"PennyLane Adam, learning rate 0.06; full-batch class-weighted binary cross entropy",
        "gradient_validation":"one-coordinate centered finite-difference check for each architecture/fold before training; all checks must pass",
        "device":"PennyLane default.qubit CPU statevector, analytic backpropagation; no GPU/hardware claim",
        "pennylane_version":qml.__version__,"runtime_seconds":time.perf_counter()-started_all,
        "available_ram_start_mib":before,"available_ram_after_import_mib":after_import,
        "available_ram_end_mib":psutil.virtual_memory().available/2**20,
        "folds":5,"origins":int(len(bug_ids)),"records":int(len(df)),
        "results":aggregate,"production_model_modified":False,"frozen_test_accessed":False}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
