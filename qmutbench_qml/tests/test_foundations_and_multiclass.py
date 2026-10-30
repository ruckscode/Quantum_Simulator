import numpy as np
import pytest

from qmutbench_qml.foundations import basis_probabilities, state_from_ry, validate_foundations
from qmutbench_qml.models.multiclass_vqc import ThreeClassVQC


def test_state_gate_measurement_and_autodiff_match_analytic_results():
    checks = validate_foundations()
    one_qubit = [row for row in checks if "theta" in row]
    assert max(row["state_max_abs_error"] for row in one_qubit) < 1e-12
    assert max(abs(row["state_norm"] - 1.0) for row in one_qubit) < 1e-12
    assert max(abs(row["z_expectation"] - row["expected_z"]) for row in one_qubit) < 1e-12
    assert max(abs(row["z_gradient"] - row["expected_z_gradient"]) for row in one_qubit) < 1e-12
    assert checks[-1]["bell_state_max_abs_error"] < 1e-12
    assert np.allclose(checks[-1]["bell_probabilities"], [.5, 0, 0, .5])


def test_basis_probabilities_require_normalized_state():
    assert np.allclose(basis_probabilities(state_from_ry(np.pi / 3)), [.75, .25])
    with pytest.raises(ValueError):
        basis_probabilities(np.array([1.0, 1.0]))


def test_three_class_qnn_updates_weights_and_returns_valid_class_probabilities():
    x = np.array([[.05, .1, .2], [.8, .9, .7], [.2, .8, .9],
                  [.1, .2, .1], [.9, .8, .7], [.25, .9, .8]])
    y = np.array([0, 1, 2, 0, 1, 2])
    model = ThreeClassVQC(n_qubits=3, depth=1, steps=2, seed=13)
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (len(x), 3)
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert not np.allclose(model.initial_weights, model.weights)
    assert model.training_executions > 0
    assert model.parameter_count == 9


def test_trainable_hybrid_readout_changes_with_supervised_loss():
    x = np.array([[.05, .1, .2, .3], [.8, .9, .7, .6], [.2, .8, .9, .4],
                  [.1, .2, .1, .3], [.9, .8, .7, .6], [.25, .9, .8, .3]])
    y = np.array([0, 1, 2, 0, 1, 2])
    model = ThreeClassVQC(steps=2, seed=17, trainable_readout=True).fit(x, y)
    assert model.parameter_count == 36
    assert model.readout_changed
    assert model.training_executions > 0
    assert model.predict_proba(x).shape == (len(x), 3)


def test_chunked_gradient_matches_exact_full_batch_update():
    x = np.array([[.05, .1, .2, .3], [.8, .9, .7, .6], [.2, .8, .9, .4],
                  [.1, .2, .1, .3], [.9, .8, .7, .6], [.25, .9, .8, .3]])
    y = np.array([0, 1, 2, 0, 1, 2])
    full = ThreeClassVQC(steps=1, seed=33, trainable_readout=True).fit(x, y)
    chunked = ThreeClassVQC(steps=1, seed=33, trainable_readout=True).fit(
        x, y, gradient_batch_size=2)
    assert np.allclose(full.weights, chunked.weights, atol=1e-10)
    assert np.allclose(full.readout_weights, chunked.readout_weights, atol=1e-10)
    assert np.allclose(full.loss_history, chunked.loss_history, atol=1e-10)
