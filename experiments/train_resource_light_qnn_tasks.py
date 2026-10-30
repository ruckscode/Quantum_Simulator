"""Train small differentiable statevector QNNs on the two pending binary tasks.

This runner avoids PennyLane/sklearn imports and keeps a 500 MiB RAM floor. It
uses outer GroupKFold-like folds, fixed hashing features, parameter-shift
gradients, and resumable per-epoch checkpoints.
"""
from __future__ import annotations

import csv, hashlib, json, math, os, re, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
FLOOR_MIB = 500
STEPS = int(os.environ.get("LIGHT_QNN_EPOCHS", "12"))
BATCH = int(os.environ.get("LIGHT_QNN_BATCH", "64"))
SEED = int(os.environ.get("LIGHT_QNN_SEED", "28102026"))


def check_ram():
    available = psutil.virtual_memory().available / 2**20
    if available < FLOOR_MIB:
        raise MemoryError(f"available RAM {available:.1f} MiB is below {FLOOR_MIB} MiB safety floor")
    return available


def features_from_text(text, dims=8):
    tokens = re.findall(r"[A-Za-z_]+|\d+|[^\s]", str(text).lower())
    values = [0.0] * dims
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        values[int.from_bytes(digest, "little") % dims] += math.log1p(count)
    return values


def grouped_folds(y, groups, n_splits=5):
    members = defaultdict(list)
    for i, group in enumerate(groups): members[str(group)].append(i)
    fold_rows = [[0, 0] for _ in range(n_splits)]
    fold_groups = [[] for _ in range(n_splits)]
    def group_key(item):
        _, idx = item
        counts = Counter(int(y[i]) for i in idx)
        return (-max(counts.values()), -len(idx), hashlib.md5(item[0].encode()).hexdigest())
    for group, idx in sorted(members.items(), key=group_key):
        counts = Counter(int(y[i]) for i in idx)
        target = min(range(n_splits), key=lambda f: (sum(fold_rows[f]), max(fold_rows[f]), len(fold_groups[f]), f))
        fold_groups[target].append(group)
        for c in (0, 1): fold_rows[target][c] += counts.get(c, 0)
    for fold in range(n_splits):
        valid_groups = set(fold_groups[fold])
        valid = [i for i, group in enumerate(groups) if str(group) in valid_groups]
        train = [i for i, group in enumerate(groups) if str(group) not in valid_groups]
        if not set(groups[i] for i in train).isdisjoint(set(groups[i] for i in valid)):
            raise AssertionError("group leakage")
        yield train, valid


def minmax_fit_apply(train, valid):
    mins = [min(row[j] for row in train) for j in range(len(train[0]))]
    maxs = [max(row[j] for row in train) for j in range(len(train[0]))]
    def transform(rows):
        return [[(row[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] > mins[j] else 0.0
                 for j in range(len(mins))] for row in rows]
    return transform(train), transform(valid), {"min": mins, "max": maxs}


def apply_ry(states, theta, wire):
    bit = 1 << wire
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    out = states.copy()
    for lo in range(states.shape[1]):
        if not lo & bit:
            hi = lo | bit
            a, b = states[:, lo].copy(), states[:, hi].copy()
            out[:, lo] = c * a - s * b
            out[:, hi] = s * a + c * b
    return out


def entangle(states, n_qubits):
    out = states.copy()
    for control in range(n_qubits):
        target = (control + 1) % n_qubits
        if target == control: continue
        perm = list(range(1 << n_qubits))
        for basis in range(1 << n_qubits):
            if (basis >> control) & 1: perm[basis] = basis ^ (1 << target)
        tmp = out.copy()
        out[:, perm] = tmp
    return out


def q_expectations(x, theta, n_qubits, encoding, depth, reupload=False):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if encoding == "amplitude":
        states = x.astype(complex)
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        states = states / np.maximum(norms, 1e-12)
    else:
        states = np.zeros((len(x), 1 << n_qubits), dtype=complex)
        states[:, 0] = 1.0
        # Batch-wise angle encoding.
        for q in range(n_qubits):
            bit = 1 << q
            c = np.cos(math.pi * x[:, q] / 2); s = np.sin(math.pi * x[:, q] / 2)
            for lo in range(1 << n_qubits):
                if not lo & bit:
                    hi = lo | bit
                    a, b = states[:, lo].copy(), states[:, hi].copy()
                    states[:, lo] = c * a - s * b
                    states[:, hi] = s * a + c * b
    p = 0
    for layer in range(depth):
        if encoding == "angle" and (layer == 0 or reupload):
            for q in range(n_qubits):
                bit = 1 << q
                c = np.cos(math.pi * x[:, q] / 2); s = np.sin(math.pi * x[:, q] / 2)
                for lo in range(1 << n_qubits):
                    if not lo & bit:
                        hi = lo | bit
                        a, b = states[:, lo].copy(), states[:, hi].copy()
                        states[:, lo] = c * a - s * b
                        states[:, hi] = s * a + c * b
        for q in range(n_qubits):
            states = apply_ry(states, float(theta[p]), q); p += 1
        states = entangle(states, n_qubits)
    probs = np.abs(states) ** 2
    z = np.empty((len(x), n_qubits), dtype=float)
    for q in range(n_qubits):
        bit = 1 << q
        z[:, q] = probs[:, [i for i in range(1 << n_qubits) if not (i & bit)]].sum(axis=1) - \
                  probs[:, [i for i in range(1 << n_qubits) if i & bit]].sum(axis=1)
    return z


def sigmoid(x):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def predict_prob(x, theta, head, bias, config):
    z = q_expectations(x, theta, config["qubits"], config["encoding"], config["depth"], config["reupload"])
    return sigmoid(z @ head + bias)


def metrics(y, pred):
    import numpy as np
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    per, recalls, f1s = {}, [], []
    for cls in (0, 1):
        tp = int(np.sum((y == cls) & (pred == cls))); fp = int(np.sum((y != cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls))); support = int(np.sum(y == cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[str(cls)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall); f1s.append(f1)
    return {"accuracy": float(np.mean(y == pred)), "balanced_accuracy": float(sum(recalls) / 2),
            "macro_f1": float(sum(f1s) / 2), "per_class": per,
            "confusion_matrix": [[int(np.sum((y == a) & (pred == b))) for b in (0, 1)] for a in (0, 1)]}


def replace_with_retry(source, destination, attempts=7):
    """Retry brief Windows/OneDrive sharing locks before failing a checkpoint write."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.05 * (2 ** attempt), 1.0))


def fit_model(x, y, config, seed, checkpoint, history_path, *, provenance=None,
              checkpoint_callback=None, sample_weights=None):
    import numpy as np
    y = np.asarray(y, dtype=float)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape != y.shape or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
            raise ValueError("sample_weights must be finite positive values aligned to y")
    n_qubits = config["qubits"]
    n_params = n_qubits * config["depth"]
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.12, n_params)
    head = rng.normal(0, 0.2, n_qubits)
    bias = 0.0
    m_t = np.zeros_like(theta); v_t = np.zeros_like(theta)
    m_h = np.zeros_like(head); v_h = np.zeros_like(head); m_b = v_b = 0.0
    lr = config["learning_rate"]
    history = []
    start_epoch=1
    if os.environ.get("LIGHT_QNN_RESUME")=="1" and checkpoint.exists():
        saved=np.load(checkpoint,allow_pickle=False)
        required={"m_t","v_t","m_h","v_h","m_b","v_b","rng_state"}
        if not required.issubset(saved.files):
            raise ValueError(f"Checkpoint {checkpoint} predates resumable optimizer state; rerun this fold with resume disabled")
        saved_config=json.loads(str(saved["config"].item()))
        if saved_config != config:
            raise ValueError(f"Checkpoint {checkpoint} model configuration is incompatible")
        if provenance is not None:
            if "provenance" not in saved.files or json.loads(str(saved["provenance"].item())) != provenance:
                raise ValueError(f"Checkpoint {checkpoint} source/task provenance is incompatible")
        theta=saved["theta"];head=saved["head"];bias=float(saved["bias"])
        m_t=saved["m_t"];v_t=saved["v_t"];m_h=saved["m_h"];v_h=saved["v_h"];m_b=saved["m_b"];v_b=saved["v_b"]
        start_epoch=int(saved["epoch"])+1;rng.bit_generator.state=json.loads(str(saved["rng_state"].item()))
        if history_path.exists():history=json.loads(history_path.read_text(encoding="utf-8"))
        saved.close()
    started = time.perf_counter()
    for epoch in range(start_epoch, STEPS + 1):
        order = rng.permutation(len(y)); losses = []
        for start in range(0, len(y), BATCH):
            check_ram()
            ix = order[start:start+BATCH]; xb = x[ix]; yb = y[ix]
            z = q_expectations(xb, theta, n_qubits, config["encoding"], config["depth"], config["reupload"])
            logits = z @ head + bias
            prob = sigmoid(logits)
            batch_weights = 1.0 if sample_weights is None else sample_weights[ix]
            dlogit = (prob - yb) * batch_weights / len(ix)
            gh = z.T @ dlogit; gb = float(dlogit.sum())
            dz = dlogit[:, None] * head[None, :]
            gt = np.zeros_like(theta)
            shift = math.pi / 2
            for j in range(n_params):
                plus = theta.copy(); plus[j] += shift
                minus = theta.copy(); minus[j] -= shift
                zp = q_expectations(xb, plus, n_qubits, config["encoding"], config["depth"], config["reupload"])
                zm = q_expectations(xb, minus, n_qubits, config["encoding"], config["depth"], config["reupload"])
                dzj = (zp - zm) / 2.0
                gt[j] = float(np.sum(dz * dzj))
            t = (epoch - 1) * math.ceil(len(y) / BATCH) + start // BATCH + 1
            beta1, beta2 = .9, .999
            m_t = beta1*m_t + (1-beta1)*gt; v_t = beta2*v_t + (1-beta2)*(gt*gt)
            m_h = beta1*m_h + (1-beta1)*gh; v_h = beta2*v_h + (1-beta2)*(gh*gh)
            m_b = beta1*m_b + (1-beta1)*gb; v_b = beta2*v_b + (1-beta2)*(gb*gb)
            theta -= lr * (m_t/(1-beta1**t)) / (np.sqrt(v_t/(1-beta2**t)) + 1e-8)
            head -= lr * (m_h/(1-beta1**t)) / (np.sqrt(v_h/(1-beta2**t)) + 1e-8)
            bias -= lr * (m_b/(1-beta1**t)) / (math.sqrt(v_b/(1-beta2**t)) + 1e-8)
            row_loss = -(yb*np.log(prob+1e-12)+(1-yb)*np.log(1-prob+1e-12))
            losses.append(float(np.mean(row_loss * batch_weights)))
        history.append({"epoch": epoch, "train_batch_loss": float(np.mean(losses)),
                        "elapsed_seconds": time.perf_counter()-started,
                        "available_ram_mib": check_ram()})
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{checkpoint.name}.", suffix=".tmp", dir=checkpoint.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                np.savez(target, theta=theta, head=head, bias=np.asarray(bias),
                         config=np.asarray(json.dumps(config, sort_keys=True)), epoch=np.asarray(epoch),
                         m_t=m_t,v_t=v_t,m_h=m_h,v_h=v_h,m_b=m_b,v_b=v_b,
                         rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
                         **({"provenance": np.asarray(json.dumps(provenance, sort_keys=True))} if provenance is not None else {}))
                target.flush(); os.fsync(target.fileno())
            replace_with_retry(temp_name, checkpoint)
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        history_data = json.dumps(history, indent=2).encode("utf-8")
        fd, temp_history = tempfile.mkstemp(prefix=f".{history_path.name}.", suffix=".tmp", dir=history_path.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(history_data); target.flush(); os.fsync(target.fileno())
            replace_with_retry(temp_history, history_path)
        finally:
            if os.path.exists(temp_history): os.unlink(temp_history)
        if checkpoint_callback is not None:
            checkpoint_callback({"epoch": epoch, "checkpoint_path": checkpoint,
                                 "history_path": history_path, "history": history})
    return theta, head, bias, history, time.perf_counter()-started


def gradient_check(config):
    import numpy as np
    x = np.ones((2, config["features"])) * .37
    theta = np.linspace(-.2, .3, config["qubits"] * config["depth"])
    head = np.linspace(.1, .3, config["qubits"]); bias = -.1
    def loss(th):
        p = np.clip(predict_prob(x, th, head, bias, config), 1e-8, 1-1e-8)
        y = np.asarray([0., 1.])
        return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    j = 0; eps = 1e-5
    plus=theta.copy(); plus[j]+=eps; minus=theta.copy(); minus[j]-=eps
    fd=(loss(plus)-loss(minus))/(2*eps)
    z=q_expectations(x,theta,config["qubits"],config["encoding"],config["depth"],config["reupload"])
    prob=sigmoid(z@head+bias); dlogit=(prob-np.asarray([0.,1.]))/2
    dz=dlogit[:,None]*head[None,:]
    tp=theta.copy(); tp[j]+=math.pi/2; tm=theta.copy(); tm[j]-=math.pi/2
    grad=float(np.sum(dz*(q_expectations(x,tp,config["qubits"],config["encoding"],config["depth"],config["reupload"])-q_expectations(x,tm,config["qubits"],config["encoding"],config["depth"],config["reupload"]))/2))
    return {"parameter_shift":grad,"finite_difference":fd,"absolute_error":abs(grad-fd),"passed":abs(grad-fd)<2e-4}


def task_run(name, manifest, group_name):
    import numpy as np
    with manifest.open(encoding="utf-8-sig", newline="") as f: rows=list(csv.DictReader(f))
    y=np.asarray([int(r["label"]) for r in rows],dtype=int)
    groups=[r[group_name] for r in rows]
    X=np.asarray([features_from_text(r["code"],8) for r in rows],dtype=float)
    if os.environ.get("LIGHT_QNN_SWEEP")=="1":
        configs=[
            {"name":"amplitude_hybrid_3q_depth1_lr004","qubits":3,"features":8,"encoding":"amplitude","depth":1,"reupload":False,"learning_rate":.04},
            {"name":"amplitude_hybrid_3q_depth2_lr004","qubits":3,"features":8,"encoding":"amplitude","depth":2,"reupload":False,"learning_rate":.04},
            {"name":"angle_reupload_4q_depth2_lr002","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":True,"learning_rate":.02},
            {"name":"angle_reupload_4q_depth2_lr0035","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":True,"learning_rate":.035},
            {"name":"angle_reupload_4q_depth2_lr007","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":True,"learning_rate":.07},
            {"name":"angle_4q_depth2_lr004","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":False,"learning_rate":.04},
        ]
    else:
        configs=[
            {"name":"amplitude_hybrid_3q_depth2","qubits":3,"features":8,"encoding":"amplitude","depth":2,"reupload":False,"learning_rate":.04},
            {"name":"angle_reupload_hybrid_4q_depth2","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":True,"learning_rate":.035},
        ]
    config_filter={v.strip() for v in os.environ.get("LIGHT_QNN_CONFIG_FILTER", "").split(",") if v.strip()}
    if config_filter:
        unknown=config_filter-{cfg["name"] for cfg in configs}
        if unknown: raise ValueError(f"unknown QNN config filter values: {sorted(unknown)}")
        configs=[cfg for cfg in configs if cfg["name"] in config_filter]
    output=CAMPAIGN/name/os.environ.get("LIGHT_QNN_OUT_DIR","resource_light_qnn"); output.mkdir(parents=True,exist_ok=True)
    all_predictions=[]; fold_summaries=[]; start_task=time.perf_counter()
    for fold,(tr,va) in enumerate(grouped_folds(y,groups),1):
        if len(set(y[tr]))<2 or len(set(y[va]))<2: raise ValueError(f"fold {fold} lacks class coverage")
        train,valid,scale=minmax_fit_apply(X[tr].tolist(),X[va].tolist())
        for config in configs:
            train_x=np.asarray(train,dtype=float)[:,:config["features"]]
            valid_x=np.asarray(valid,dtype=float)[:,:config["features"]]
            # Amplitude state has dimension 2^n; deterministic hashed counts are padded/truncated.
            if config["encoding"]=="amplitude":
                train_x=np.pad(train_x,((0,0),(0,max(0,2**config["qubits"]-train_x.shape[1]))))[:,:2**config["qubits"]]
                valid_x=np.pad(valid_x,((0,0),(0,max(0,2**config["qubits"]-valid_x.shape[1]))))[:,:2**config["qubits"]]
            gc=gradient_check(config)
            if not gc["passed"]: raise ArithmeticError(f"gradient check failed: {config['name']} {gc}")
            folder=output/config["name"]/f"fold_{fold}"; folder.mkdir(parents=True,exist_ok=True)
            theta,head,bias,history,seconds=fit_model(train_x,y[tr].astype(float),config,SEED+fold,
                folder/"checkpoint.npz",folder/"training_history.json")
            prob=predict_prob(valid_x,theta,head,bias,config)
            pred=(prob>=.5).astype(int); report=metrics(y[va],pred)
            record={"task":name,"model":config["name"],"fold":fold,"records":len(va),"groups":len(set(groups[i] for i in va)),
                    "training_seconds":seconds,"gradient_check":gc,"metrics":report,"available_ram_mib":check_ram()}
            fold_summaries.append(record)
            for j,i in enumerate(va): all_predictions.append({"record_id":rows[i]["record_id"],"group_id":str(groups[i]),"fold":fold,
                "model":config["name"],"y_true":int(y[i]),"probability":float(prob[j]),"y_pred":int(pred[j])})
            (folder/"fold_result.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    result_by_model={}
    for config in configs:
        subset=[r for r in all_predictions if r["model"]==config["name"]]
        score=metrics([r["y_true"] for r in subset],[r["y_pred"] for r in subset])
        result_by_model[config["name"]]={**score,"folds":5,"records":len(subset),
            "training_seconds":sum(r["training_seconds"] for r in fold_summaries if r["model"]==config["name"]),
            "architecture":config,"checkpoint_paths":[str((output/config["name"]/f"fold_{i}"/"checkpoint.npz").relative_to(ROOT)) for i in range(1,6)]}
    with (output/"oof_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_predictions[0]));w.writeheader();w.writerows(all_predictions)
    (output/"summary.json").write_text(json.dumps({"status":"completed_resource_light_grouped_qnn","task":name,
        "records":len(rows),"groups":len(set(groups)),"folds":5,"features":"8-bin BLAKE2b hashed token counts, train-fold min-max scaling",
        "split":"five deterministic group-disjoint folds; all paired/source variants held by bug_id",
        "results":result_by_model,"fold_results":fold_summaries,"optimizer_steps_per_fold":STEPS,"batch_size":BATCH,
        "runtime_seconds":time.perf_counter()-start_task,
        "minimum_ram_floor_mib":FLOOR_MIB,"frozen_test_accessed":False},indent=2),encoding="utf-8")
    print(json.dumps({"task":name,"results":result_by_model},indent=2))


def main():
    if check_ram()<700: raise MemoryError("QNN task preflight requires 700 MiB available")
    import numpy as np
    if check_ram()<650: raise MemoryError("NumPy import left less than 650 MiB available; stopping before training")
    tasks=[("bugs4q_robust_execution","bugs4q_robust_execution/normalized_manifest.csv","bug_id"),
           ("qsimbench_backend_fingerprint","qsimbench_backend_fingerprint/normalized_manifest.csv","bug_id")]
    selected={v.strip() for v in os.environ.get("LIGHT_QNN_TASK_FILTER", "").split(",") if v.strip()}
    if selected:
        unknown=selected-{name for name,_,_ in tasks}
        if unknown: raise ValueError(f"unknown task filter values: {sorted(unknown)}")
        tasks=[task for task in tasks if task[0] in selected]
    for name,relative,group in tasks:
        task_run(name,CAMPAIGN/relative,group)


if __name__=="__main__": main()
