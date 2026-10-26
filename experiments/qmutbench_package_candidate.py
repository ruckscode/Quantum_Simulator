"""Fit and package the confirmed mutation-aware QMutBench model."""
from __future__ import annotations

import gc, hashlib, json, platform, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from experiments.qmutbench_exhaustive_campaign import BASE_FEATURES, FEATURES, OUT, load_data, model
from experiments.qmutbench_qasm_model_comparison import SEED
from experiments.qmutbench_qasm_features import SOURCE_CSV, sha256_file

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"data/qmutbench_ml/mutation_aware_v1"
META_SOURCE=ROOT/"data/qmutbench_ml/position_fraction_v1/best_model_metadata.json"

def run():
    x,oof=load_data(); names=BASE_FEATURES+FEATURES
    BUNDLE.mkdir(parents=True,exist_ok=True)
    estimator=model(names,0,100)
    start=time.perf_counter()
    with threadpool_limits(limits=1): estimator.fit(x[names],oof.y.to_numpy(np.int8))
    model_path=BUNDLE/"best_model.joblib"
    joblib.dump(estimator,model_path,compress=3)
    del estimator; gc.collect()
    previous=json.loads(META_SOURCE.read_text(encoding="utf-8"))
    qasm_inputs=["original_qasm","mutant_qasm"]
    metadata={
        "schema":"qmutbench-inference-model-v1","model_version":"qmutbench-mutation-aware-v1",
        "model_path":"data/qmutbench_ml/mutation_aware_v1/best_model.joblib",
        "model_sha256":sha256_file(model_path),"feature_schema":names,
        "input_feature_schema":previous["input_feature_schema"]+qasm_inputs,
        "feature_derivations":{"position_fraction":previous["feature_derivations"]["position_fraction"]},
        "qasm_feature_derivation":{"original_column":"original_qasm","mutant_column":"mutant_qasm",
            "operation_column":"operation","gate_column":"gate","output_features":FEATURES,
            "source_format":"OpenQASM 2.0 text","definition":"common-prefix/common-suffix edit alignment on gate-name and wire-tuple sequences; measurement and barrier instructions excluded"},
        "inference_threshold":0.5,"threshold_selection":"fixed 0.5; no calibration or threshold adjustment",
        "target_map":{"OPO":1,"Survivor":0,"WOO":1},
        "training_scope":"development OOF rows and origins only; historical frozen test excluded",
        "training_rows":len(oof),"training_origin_groups":int(oof.origin_program.nunique()),
        "training_data_sha256":sha256_file(SOURCE_CSV),
        "development_oof_sha256":sha256_file(ROOT/"data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"),
        "fixed_three_fold_confirmation_sha256":sha256_file(OUT/"mutation_diff_screen.json"),
        "five_fold_grouped_confirmation_sha256":sha256_file(OUT/"five_fold_confirmation/summary.json"),
        "qasm_feature_cache_manifest_sha256":sha256_file(ROOT/"data/qmutbench_ml/qasm_mutation_diff_v1/manifest.json"),
        "model_parameters":{"n_estimators":100,"max_depth":22,"min_samples_leaf":15,"max_features":"sqrt","class_weight":"balanced_subsample","n_jobs":1,"random_state":SEED},
        "fit_seconds":time.perf_counter()-start,"python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__,
        "note":"OOF probabilities are uncalibrated model scores; the 85% aspirational target was not reached. Each prediction requires original and mutant OpenQASM 2.0 source text."
    }
    meta=BUNDLE/"best_model_metadata.json"
    meta.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"model_path":str(model_path),"metadata_path":str(meta),"model_sha256":metadata["model_sha256"],"fit_seconds":metadata["fit_seconds"],"feature_count":len(names)},indent=2))

if __name__=="__main__": run()
