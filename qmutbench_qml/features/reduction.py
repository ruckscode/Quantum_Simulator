"""Fold-local encoding, scaling, and train-only feature selection."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from qmutbench_qml.data.loader import CATEGORICAL, NUMERIC


class ReducedFeatureEncoder:
    """Select four model-input scalars using only the current training fold."""

    def __init__(self, n_features: int = 4, seed: int = 20261026):
        self.n_features = n_features
        self.seed = seed

    def fit(self, frame, y):
        self.transformer = ColumnTransformer([
            ("numeric", MinMaxScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ], sparse_threshold=0.0, verbose_feature_names_out=True)
        encoded = self.transformer.fit_transform(frame)
        self.names = self.transformer.get_feature_names_out()
        discrete = np.array([name.startswith("categorical__") for name in self.names])
        scores = mutual_info_classif(encoded, np.asarray(y, dtype=int), discrete_features=discrete,
                                     random_state=self.seed)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(self.names[i])))
        self.selected_indices = np.asarray(ranked[: self.n_features], dtype=int)
        self.selected_names = [str(self.names[i]) for i in self.selected_indices]
        reduced = encoded[:, self.selected_indices]
        self.angle_scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
        self.angle_scaler.fit(reduced)
        return self

    def transform(self, frame):
        encoded = self.transformer.transform(frame)
        return self.angle_scaler.transform(encoded[:, self.selected_indices])


def feature_parity_table():
    """Explain how all 23 incumbent inputs are treated in this small-wire pilot."""
    rows = []
    for name in ["algorithm", "operation", "gate", "dominant_state"]:
        rows.append({"incumbent_feature": name, "type": "categorical", "pilot_handling": "fold-local one-hot encoding; candidate dummy columns compete in train-only MI selection", "exclusion": "not always selected among four encoded scalars; full vocabularies expand beyond four input wires"})
    for name in ["qubits", "gates", "Position_int", "position_fraction", "qasm_diff_orig_span", "qasm_diff_mut_span", "qasm_diff_total_span", "qasm_diff_expected_gate_match", "qasm_diff_qubit_set_equal", "qasm_diff_qubits_touched", "qasm_diff_removed_arity", "qasm_diff_added_arity", "qasm_diff_exact_single_edit", "qasm_diff_is_local"]:
        rows.append({"incumbent_feature": name, "type": "numeric", "pilot_handling": "fold-local min-max encoding to [0,1], then train-only MI selection", "exclusion": "not selected if below top four because the pilot has four input wires"})
    for name in ["qasm_diff_removed_gate", "qasm_diff_added_gate", "qasm_diff_alignment_class", "qasm_diff_local_prev_gate", "qasm_diff_local_next_gate"]:
        rows.append({"incumbent_feature": name, "type": "categorical", "pilot_handling": "fold-local one-hot encoding; candidate dummy columns compete in train-only MI selection", "exclusion": "not always selected among four encoded scalars; full vocabularies expand beyond four input wires"})
    rows += [
        {"incumbent_feature": "position", "type": "categorical source band", "pilot_handling": "replaced by the same saved position_fraction midpoint mapping used by incumbent model", "exclusion": "source band is not separately encoded in the 23-feature incumbent schema"},
        {"incumbent_feature": "original_qasm / mutant_qasm", "type": "OpenQASM 2.0 source", "pilot_handling": "reuse the incumbent's cached, outcome-independent mutation-diff features", "exclusion": "raw token/gate sequences are too long and variable-length for the initial four-wire feature map"},
        {"incumbent_feature": "Killed", "type": "target", "pilot_handling": "used only to verify the source target mapping and construct y", "exclusion": "never a predictor"},
        {"incumbent_feature": "Origin_program", "type": "group identifier", "pilot_handling": "used only for origin-disjoint grouped evaluation", "exclusion": "never a predictor; avoids sibling-record leakage"},
        {"incumbent_feature": "Unnamed: 0 / record_id", "type": "row identifier", "pilot_handling": "used only for record alignment", "exclusion": "never a predictor; nonsemantic row identity could leak ordering"},
    ]
    return rows
