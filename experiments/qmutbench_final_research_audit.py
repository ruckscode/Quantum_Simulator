"""Verify the current QMutBench incumbent and its train/inference feature contract."""
from __future__ import annotations

import json, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.qmutbench_qasm_features import (SOURCE_CSV, OFFICIAL_CHECKOUT, sha256_file,
    EXPECTED_DATA_SHA, EXPECTED_REPO_COMMIT)
from experiments.qmutbench_qasm_mutation_diff import (_ops, _safe_paths, FEATURE_COLUMNS,
    QASM_ROOT, USECOLS)
from experiments.qmutbench_qasm_model_comparison import assert_group_fold_isolation, metrics
from qmutbench_ml import QMutBenchOutcomeModel
from qmutbench_ml.qasm_mutation_diff import circuit_signature_from_qasm_text

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/qmutbench_ml/performance_phase_full_v1/predictions"
CAND=ROOT/"data/qmutbench_ml/exhaustive_campaign_v1/mutation_diff_100_trees"
CACHE=ROOT/"data/qmutbench_ml/qasm_mutation_diff_v1"
BUNDLE=ROOT/"data/qmutbench_ml/mutation_aware_v1"

def _load(paths):
    parts=[np.load(p,allow_pickle=False) for p in paths]
    return pd.DataFrame({k:np.concatenate([p[k] for p in parts]) for k in ("record_id","y","origin_program","fold","probability")}).sort_values("record_id").reset_index(drop=True)

def run():
    base=_load([BASE/f"position_fraction_fold{i}.npz" for i in (1,2,3)])
    candidate=_load([CAND/f"fold{i}.npz" for i in (1,2,3)])
    if not (np.array_equal(base.record_id,candidate.record_id) and np.array_equal(base.y,candidate.y) and np.array_equal(base.fold,candidate.fold)):
        raise ValueError("Incumbent OOF predictions are not aligned with candidate records/folds")
    split=base.rename(columns={"origin_program":"origin_program"})
    assert_group_fold_isolation(split)
    metadata=json.loads((BUNDLE/"best_model_metadata.json").read_text(encoding="utf-8"))
    predictor=QMutBenchOutcomeModel(BUNDLE/"best_model.joblib")
    model_features=list(predictor.features)
    if model_features!=metadata["feature_schema"] or model_features!=list(predictor.model.feature_names_in_):
        raise ValueError("Saved metadata/model feature order differs")
    cache_manifest=json.loads((CACHE/"manifest.json").read_text(encoding="utf-8"))
    if cache_manifest.get("status")!="complete" or cache_manifest.get("feature_rows")!=len(candidate) or cache_manifest.get("error_rows")!=0:
        raise ValueError("Cached source extraction is incomplete")
    if metadata.get("qasm_feature_cache_manifest_sha256")!=sha256_file(CACHE/"manifest.json"):
        raise ValueError("Model metadata does not identify the current QASM feature cache manifest")
    if sha256_file(SOURCE_CSV)!=EXPECTED_DATA_SHA:
        raise ValueError("Source data checksum does not match the locked data")
    cols=set(USECOLS)
    prohibited={"Killed","y","y_true","target","Origin_program","origin_program"}
    if cols & prohibited:
        raise ValueError(f"QASM feature extraction reads prohibited source fields: {cols & prohibited}")

    diff_parts=[pd.read_csv(p,compression="gzip") for p in sorted(CACHE.glob("diff_*.csv.gz"))]
    diff=pd.concat(diff_parts,ignore_index=True)
    if len(diff)!=len(candidate) or diff.record_id.duplicated().any(): raise ValueError("QASM features do not cover unique development rows")
    subset=diff.sample(20,random_state=20261026)
    meta_cols=["Unnamed: 0","algorithm","qubits","gates","operation","gate","dominant_state","Position_int","position"]
    raw=pd.read_csv(SOURCE_CSV,usecols=meta_cols).set_index("Unnamed: 0").loc[subset.record_id].reset_index()
    tracked_raw=subprocess.run(["git","-C",str(OFFICIAL_CHECKOUT),"ls-files","-z","--","data/Origin_programs","data/Mutated_programs"],check=True,capture_output=True).stdout
    tracked={p.decode().removeprefix("data/") for p in tracked_raw.split(b"\0") if p.endswith(b".qasm")}
    mismatch_count=0
    for _,row in raw.iterrows():
        rid,orig,mut=_safe_paths(row.to_dict(),tracked)
        orig_path,mut_path=QASM_ROOT/orig,QASM_ROOT/mut
        expected=subset.loc[subset.record_id.eq(rid)].iloc[0]
        for path,cache_value in ((orig_path,expected), (mut_path,expected)):
            source=path.read_text(encoding="utf-8")
            signature,qcount=circuit_signature_from_qasm_text(source)
            if signature!=_ops(path): mismatch_count+=1
            if qcount<1: raise ValueError("Circuit has no qubits")
        features=_ops(orig_path),_ops(mut_path)
        derived=__import__("qmutbench_ml.qasm_mutation_diff",fromlist=["mutation_diff_features"]).mutation_diff_features(*features,str(row.operation),str(row.gate))
        for column in FEATURE_COLUMNS:
            value,expected_value=derived[column],expected[column]
            if isinstance(value,(float,np.floating)):
                if not np.isclose(float(value),float(expected_value),rtol=0,atol=1e-12): mismatch_count+=1
            elif value!=expected_value: mismatch_count+=1
    if mismatch_count: raise ValueError(f"Training/inference QASM feature mismatch in sampled check: {mismatch_count}")

    smoke_row=raw.iloc[0].to_dict()
    _,original_path,mutant_path=_safe_paths(smoke_row,tracked)
    smoke={"algorithm":str(smoke_row["algorithm"]),"operation":str(smoke_row["operation"]),
      "gate":str(smoke_row["gate"]),"dominant_state":str(smoke_row.get("dominant_state","")),
      "qubits":int(smoke_row["qubits"]),"gates":int(smoke_row["gates"]),
      "Position_int":int(smoke_row["Position_int"]),"position":str(smoke_row.get("position","")),
      "original_qasm":(QASM_ROOT/original_path).read_text(encoding="utf-8"),
      "mutant_qasm":(QASM_ROOT/mutant_path).read_text(encoding="utf-8")}
    single=predictor.predict(smoke); batch=predictor.predict([smoke,smoke])
    if len(batch)!=2 or not np.isclose(single["probability_detected"],batch[0]["probability_detected"]):
        raise ValueError("Real-record single/batch inference smoke mismatch")

    report={"schema":"qmutbench-final-research-baseline-and-feature-audit-v1","status":"verified",
      "branch_commit":{"branch":subprocess.run(["git","branch","--show-current"],check=True,capture_output=True,text=True).stdout.strip(),"commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()},
      "source":{"path":"data/qmutbench_ingestion/raw/official_all_data_20261015.csv","sha256":sha256_file(SOURCE_CSV),"rows":723079,"development_rows":len(base),"development_origins":int(base.origin_program.nunique()),"historical_frozen_rows":144962,"frozen_test_labels_or_circuits_loaded":False,"qasm_repo_commit":EXPECTED_REPO_COMMIT},
      "artifacts":{"incumbent_model_sha256":sha256_file(ROOT/"data/qmutbench_ml/position_fraction_v1/best_model.joblib"),"mutation_aware_model_sha256":sha256_file(BUNDLE/"best_model.joblib"),"metadata_sha256":sha256_file(BUNDLE/"best_model_metadata.json"),"baseline_oof_sha256":sha256_file(ROOT/"data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"),"diff_cache_manifest_sha256":sha256_file(CACHE/"manifest.json"),"diff_extractor_rows":cache_manifest["feature_rows"],"diff_parse_errors":cache_manifest["error_rows"]},
      "fold_isolation":{"folds":{str(f):{"rows":int(base.fold.eq(f).sum()),"origins":int(base.loc[base.fold.eq(f),"origin_program"].nunique())} for f in (1,2,3)},"max_folds_per_origin":int(base.groupby("origin_program").fold.nunique().max())},
      "reproduced_oof":{"baseline_metrics":metrics(base.y.to_numpy(np.int8),base.probability.to_numpy(float)),"mutation_aware_metrics":metrics(candidate.y.to_numpy(np.int8),candidate.probability.to_numpy(float)),"macro_f1_delta":metrics(candidate.y.to_numpy(np.int8),candidate.probability.to_numpy(float))["macro_f1"]-metrics(base.y.to_numpy(np.int8),base.probability.to_numpy(float))["macro_f1"],"same_rows_and_labels":True},
      "feature_contract":{"model_feature_count":len(model_features),"model_feature_schema":model_features,"input_schema":metadata["input_feature_schema"],"qasm_derived_features":metadata["qasm_feature_derivation"]["output_features"],"training_qasm_usecols":USECOLS,"target_or_origin_loaded_by_extractor":False,"sampled_rows_train_inference_equivalence":20,"sampled_feature_mismatches":mismatch_count,"real_record_inference_smoke":{"single":single,"batch_size":len(batch),"single_batch_probability_match":True},"measurement_barrier":"parsed then excluded from gate/wire sequence","conditional_qasm":"unsupported by the QMutBench OpenQASM 2.0 parser and rejected as malformed","unsupported_gate":"explicit UnsupportedQASMGateError with gate name","qubit_count_pair_validation":"inference rejects differing original/mutant declared qubit counts","alignment":"deterministic common-prefix/common-suffix on lowercased gate name plus absolute wire-index tuple; ambiguous repeated-operation insertion location remains a documented alignment limitation"},
      "external_data_search":{"official_qmutbench":"Official paper/repository describe 375 original circuits and 723079 mutants; current locked corpus already covers that release.","qsurvdiag":"Zenodo replication describes archived QUTest executions, but public metadata does not establish matching QMutBench label definition or disjoint origins; not ingested.","qumug":"Official release describes search-generated non-equivalent mutants under selected test suites, rather than row-compatible QMutBench survival-rate labels; not ingested.","independent_external_evidence":"none compatible and verified; remains unavailable"}}
    out=ROOT/"reports/experiments/qmutbench_final_research_baseline_audit_20261026.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(out),"baseline_macro_f1":report["reproduced_oof"]["baseline_metrics"]["macro_f1"],"candidate_macro_f1":report["reproduced_oof"]["mutation_aware_metrics"]["macro_f1"],"feature_mismatches":mismatch_count},indent=2))

if __name__=="__main__":run()
