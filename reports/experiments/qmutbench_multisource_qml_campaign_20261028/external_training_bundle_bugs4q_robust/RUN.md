# Run from the extracted bundle root on Linux (Kaggle or equivalent).
python -m pip install pennylane==0.45.1 scikit-learn==1.9.1 pandas==3.0.6 psutil==7.2.2
export BUGS4Q_QNN_INPUT="$PWD/workspace/reports/bugs4q_robust_execution/normalized_manifest.csv"
export BUGS4Q_QNN_OUTPUT="$PWD/output/bugs4q_robust_execution/qnn"
export BUGS4Q_TASK="Bugs4Q-Robust pass/fail prediction from the test harness and program snapshot"
export QNN_DATA_DESCRIPTION="2,220 program-run observations from 37 issue groups"
export QNN_GROUP_SPLIT_DESCRIPTION="5-fold GroupKFold by issue id; all snapshots and runs for an issue stay together"
export QNN_BATCH_ROWS=1 QNN_VALIDATION_TRACK_ROWS=16 QNN_TFIDF_MAX_FEATURES=512 QNN_SVD_ITERATIONS=2
mkdir -p "$BUGS4Q_QNN_OUTPUT"
python workspace/experiments/bugs4q_qnn_grouped_benchmark.py
python workspace/experiments/finalize_bugs4q_qnn_grouped_benchmark.py
