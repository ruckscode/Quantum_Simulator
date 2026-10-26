"""Streamlit showcase for QMutBench classifier experiments.

The metrics below are copied from the checked-in research report. Real inference
is enabled only when the verified local model bundle is present.
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "experiments" / "qmutbench_final_research_report_20261026.json"
MODEL = ROOT / "data" / "qmutbench_ml" / "mutation_aware_v1" / "best_model.joblib"
METADATA = ROOT / "data" / "qmutbench_ml" / "mutation_aware_v1" / "best_model_metadata.json"

st.set_page_config(page_title="QMutBench ML Classifier Lab", page_icon="⚛️", layout="wide")
st.title("QMutBench ML Classifier Lab")
st.caption("Historical mutation-outcome classification • Research showcase")

st.info(
    "This dashboard separates recorded evaluation results from live model inference. "
    "The trained model bundle is intentionally not committed to Git; predictions are "
    "disabled until the matching model and metadata files are supplied locally."
)

# Report values are explicit and versioned, so the dashboard remains usable without
# the large, local-only .joblib artifact.
baseline = {
    "Accuracy": 76.83,
    "Balanced accuracy": 76.97,
    "Macro F1": 76.82,
    "ROC-AUC": 86.25,
    "Brier score": 0.1558,
}
mutation_aware = {
    "Accuracy": 78.89,
    "Balanced accuracy": 78.98,
    "Macro F1": 78.89,
    "ROC-AUC": 87.88,
    "Brier score": 0.1467,
}
nested = {
    "Accuracy": 79.31,
    "Balanced accuracy": 79.27,
    "Macro F1": 79.28,
    "ROC-AUC": 88.22,
    "Brier score": 0.1430,
}

try:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    oof = report["baseline_and_feature_audit"]["reproduced_oof"]
    baseline = {
        "Accuracy": 100 * oof["baseline_metrics"]["accuracy"],
        "Balanced accuracy": 100 * oof["baseline_metrics"]["balanced_accuracy"],
        "Macro F1": 100 * oof["baseline_metrics"]["macro_f1"],
        "ROC-AUC": 100 * oof["baseline_metrics"]["roc_auc"],
        "Brier score": oof["baseline_metrics"]["brier_score"],
    }
    mutation_aware = {
        "Accuracy": 100 * oof["mutation_aware_metrics"]["accuracy"],
        "Balanced accuracy": 100 * oof["mutation_aware_metrics"]["balanced_accuracy"],
        "Macro F1": 100 * oof["mutation_aware_metrics"]["macro_f1"],
        "ROC-AUC": 100 * oof["mutation_aware_metrics"]["roc_auc"],
        "Brier score": oof["mutation_aware_metrics"]["brier_score"],
    }
    nm = report["nested_grouped_evaluation"]["nested_selected_metrics"]
    nested = {
        "Accuracy": 100 * nm["accuracy"],
        "Balanced accuracy": 100 * nm["balanced_accuracy"],
        "Macro F1": 100 * nm["macro_f1"],
        "ROC-AUC": 100 * nm["roc_auc"],
        "Brier score": nm["brier"],
    }
except (OSError, ValueError, KeyError, TypeError):
    st.warning("Research report JSON not found/readable; showing the report values bundled with this demo.")

a, b, c = st.columns(3)
a.metric("Development rows", "578,117")
b.metric("Original-circuit groups", "305")
c.metric("Mutation-aware macro F1", f'{mutation_aware["Macro F1"]:.2f}%')

st.subheader("Classifier comparison")
st.caption("Out-of-fold development comparison. Percentages are shown for classification metrics; lower Brier score is better.")
metric = st.selectbox("Metric", ["Macro F1", "Accuracy", "Balanced accuracy", "ROC-AUC", "Brier score"])
chart_data = pd.DataFrame({
    "Classifier / evaluation": ["Metadata baseline (OOF)", "Mutation-aware Extra Trees (OOF)", "Nested grouped selection (development)"],
    metric: [baseline[metric], mutation_aware[metric], nested[metric]],
}).set_index("Classifier / evaluation")
st.bar_chart(chart_data)

rows = []
for name, values in [
    ("Metadata baseline — OOF", baseline),
    ("Mutation-aware Extra Trees — OOF", mutation_aware),
    ("Nested grouped model selection — development", nested),
]:
    rows.append({"Evaluation": name, **values})
st.dataframe(pd.DataFrame(rows).set_index("Evaluation").style.format({
    "Accuracy": "{:.2f}%", "Balanced accuracy": "{:.2f}%",
    "Macro F1": "{:.2f}%", "ROC-AUC": "{:.2f}%", "Brier score": "{:.4f}"
}), use_container_width=True)

delta = mutation_aware["Macro F1"] - baseline["Macro F1"]
st.success(f"Mutation-aware features improved reproduced OOF macro F1 by {delta:.2f} percentage points versus the metadata baseline.")

with st.expander("Evaluation details and interpretation"):
    st.markdown(
        """
        - **Target:** historical QMutBench outcome label (Detected vs Survivor), not hardware fault diagnosis.
        - **Split discipline:** grouped by original circuit family to reduce leakage between related mutants.
        - **Mutation-aware features:** derived by comparing original and mutated OpenQASM gate/wire sequences.
        - **Nested grouped result:** development-only model-selection estimate; it is not independent external validation.
        - **Limitations:** the 85% macro-F1 target was not reached, and compatible independent external evidence is unavailable.
        - **Probability caution:** model probabilities are not calibrated confidence estimates.
        """
    )

st.divider()
st.subheader("Live prediction playground")
if not (MODEL.is_file() and METADATA.is_file()):
    st.warning(
        "Live inference unavailable: the verified model bundle is missing. "
        "This section will unlock automatically after both local files are restored."
    )
    st.code(
        "data/qmutbench_ml/mutation_aware_v1/best_model.joblib\n"
        "data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json",
        language="text",
    )
else:
    st.caption("The local bundle exists. Enter one QMutBench record to run the saved model.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        algorithm = col1.text_input("Algorithm", "qft")
        operation = col2.selectbox("Mutation operation", ["Add", "Replace", "Remove"])
        gate = col3.text_input("Gate", "h")
        col4, col5, col6 = st.columns(3)
        dominant_state = col4.text_input("Dominant state", "0")
        qubits = col5.number_input("Qubits", min_value=1, max_value=100, value=2)
        gates = col6.number_input("Gate count", min_value=1, max_value=10000, value=3)
        position_int = st.number_input("Position index", min_value=0, value=1)
        position = st.selectbox("Position band", [
            "0-10%", "11-20%", "21-30%", "31-40%", "41-50%",
            "51-60%", "61-70%", "71-80%", "81-90%", "91-100%"
        ])
        original_qasm = st.text_area(
            "Original OpenQASM 2.0",
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];',
        )
        mutant_qasm = st.text_area(
            "Mutant OpenQASM 2.0",
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; x q[0]; cx q[0],q[1];',
        )
        submitted = st.form_submit_button("Predict historical outcome", type="primary")
    if submitted:
        try:
            import sys
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from qmutbench_ml import QMutBenchOutcomeModel
            predictor = QMutBenchOutcomeModel(MODEL, METADATA)
            result = predictor.predict({
                "algorithm": algorithm,
                "operation": operation,
                "gate": gate,
                "dominant_state": dominant_state,
                "qubits": qubits,
                "gates": gates,
                "Position_int": position_int,
                "position": position,
                "original_qasm": original_qasm,
                "mutant_qasm": mutant_qasm,
            })
            st.subheader(f'Result: {result["predicted_label"]}')
            p, q, r = st.columns(3)
            p.metric("Predicted label", result["predicted_label"])
            q.metric("Probability detected", f'{100 * result["probability_detected"]:.1f}%')
            r.metric("Decision threshold", f'{result["applied_threshold"]:.2f}')
            st.caption(f'Model version: {result["model_version"]}. Probability is uncalibrated.')
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.caption("Source: reports/experiments/qmutbench_final_research_report_20261026.json • QMutBench research branch")
