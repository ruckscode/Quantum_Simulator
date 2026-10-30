"""Trainable three-class variational quantum classifier on analytic statevector."""
from __future__ import annotations

import time

import pennylane as qml
import psutil
from pennylane import numpy as pnp

RAM_STOP_MIB = 1024


class ThreeClassVQC:
    """End-to-end QNN: three measured Pauli expectations form softmax logits."""

    def __init__(self, n_qubits=4, depth=2, steps=10, learning_rate=0.04,
                 seed=20261027, data_reuploading=True, encoding="angle",
                 trainable_readout=False, device_name="default.qubit",
                 diff_method="backprop"):
        if encoding not in {"angle", "amplitude"}:
            raise ValueError("encoding must be 'angle' or 'amplitude'")
        if encoding == "angle" and n_qubits < 3:
            raise ValueError("Three readout observables require at least three qubits")
        if encoding == "amplitude" and n_qubits != 2:
            raise ValueError("Four reduced values use exactly two amplitude-encoding qubits")
        self.n_qubits = n_qubits
        self.depth = depth
        self.steps = steps
        self.learning_rate = learning_rate
        self.seed = seed
        self.data_reuploading = data_reuploading
        self.encoding = encoding
        self.trainable_readout = trainable_readout
        self.device_name = device_name
        self.diff_method = diff_method
        self.device = qml.device(device_name, wires=n_qubits, shots=None, seed=seed)

        @qml.qnode(self.device, interface="autograd", diff_method=diff_method)
        def circuit(features, weights):
            if encoding == "amplitude":
                qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
            for layer in range(depth):
                if encoding == "angle" and (layer == 0 or data_reuploading):
                    qml.AngleEmbedding(pnp.pi * features, wires=range(n_qubits), rotation="Y")
                for wire in range(n_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            if encoding == "amplitude":
                return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)),
                        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(3))

        self.circuit = circuit

    @staticmethod
    def logits_from_expectations(expectations):
        return pnp.stack(expectations, axis=-1)

    @staticmethod
    def probabilities_from_logits(logits):
        shifted = logits - pnp.max(logits, axis=-1, keepdims=True)
        values = pnp.exp(shifted)
        return values / pnp.sum(values, axis=-1, keepdims=True)

    @staticmethod
    def cross_entropy(logits, labels, sample_weights=None):
        probs = ThreeClassVQC.probabilities_from_logits(logits)
        chosen = probs[pnp.arange(len(labels)), labels]
        losses = -pnp.log(pnp.clip(chosen, 1e-9, 1.0))
        if sample_weights is not None:
            losses = losses * sample_weights
        return pnp.mean(losses)

    def _logits(self, x, weights):
        return pnp.stack([pnp.stack(self.circuit(row, weights)) for row in x])

    def fit(self, x, y, gradient_batch_size=None):
        import numpy as np
        from autograd import value_and_grad

        x = pnp.array(x, requires_grad=False)
        y = pnp.array(y, requires_grad=False, dtype=int)
        if set(np.asarray(y).tolist()) != {0, 1, 2}:
            raise ValueError("Training data must contain all three encoded classes 0, 1, and 2")
        counts = pnp.array([pnp.sum(y == c) for c in range(3)], requires_grad=False)
        class_weights = len(y) / (3.0 * counts)
        sample_weights = class_weights[y]
        rng = np.random.default_rng(self.seed)
        weights = pnp.array(rng.normal(0.0, 0.12, (self.depth, self.n_qubits, 3)), requires_grad=True)
        readout = pnp.array(np.eye(3), requires_grad=True)
        bias = pnp.array(np.zeros(3), requires_grad=True)
        self.initial_weights = np.asarray(weights).copy()
        self.initial_readout = np.asarray(readout).copy()
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        def objective(params, head, offset, x_batch=None, y_batch=None, weight_batch=None):
            features = x if x_batch is None else x_batch
            labels = y if y_batch is None else y_batch
            row_weights = sample_weights if weight_batch is None else weight_batch
            logits = self._logits(features, params)
            if self.trainable_readout:
                logits = logits @ head.T + offset
            return self.cross_entropy(logits, labels, row_weights)

        self.loss_history = []
        started = time.perf_counter()
        with qml.Tracker(self.device) as tracker:
            for _ in range(self.steps):
                if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB:
                    raise MemoryError(
                        f"Available system memory fell below the {RAM_STOP_MIB} MiB QML safety floor"
                    )
                if gradient_batch_size is not None and gradient_batch_size < len(x):
                    accumulated = None
                    cost = 0.0
                    value_grad = value_and_grad(objective, argnum=(0, 1, 2))
                    for start in range(0, len(x), gradient_batch_size):
                        stop = min(start + gradient_batch_size, len(x))
                        fraction = (stop - start) / len(x)
                        loss, gradient = value_grad(weights, readout, bias,
                            x[start:stop], y[start:stop], sample_weights[start:stop])
                        cost += float(loss) * fraction
                        if accumulated is None:
                            accumulated = [np.asarray(part, dtype=float) * fraction for part in gradient]
                        else:
                            for i, part in enumerate(gradient):
                                accumulated[i] += np.asarray(part, dtype=float) * fraction
                    gradients = tuple(pnp.array(part, requires_grad=False) for part in accumulated)
                    weights, readout, bias = optimizer.apply_grad(gradients, (weights, readout, bias))
                elif self.trainable_readout:
                    updated, cost = optimizer.step_and_cost(objective, weights, readout, bias)
                    weights, readout, bias = updated
                else:
                    weights, cost = optimizer.step_and_cost(objective, weights, readout, bias)
                    # Keep the unused identity/zero readout fixed in the plain VQC.
                    weights, _, _ = weights
                self.loss_history.append(float(cost))
        self.training_seconds = time.perf_counter() - started
        self.training_executions = int(tracker.totals.get("executions", 0))
        self.weights = weights
        self.readout_weights = readout
        self.readout_bias = bias
        self.parameter_count = int(np.asarray(weights).size + (np.asarray(readout).size + np.asarray(bias).size
                                    if self.trainable_readout else 0))
        if self.trainable_readout:
            self.readout_changed = bool(not np.allclose(self.initial_readout, readout))
        else:
            self.readout_changed = False
        return self

    def predict_proba(self, x):
        import numpy as np

        started = time.perf_counter()
        with qml.Tracker(self.device) as tracker:
            logits = np.asarray(self._logits(pnp.array(x, requires_grad=False), self.weights), dtype=float)
        if self.trainable_readout:
            logits = logits @ np.asarray(self.readout_weights).T + np.asarray(self.readout_bias)
        self.inference_seconds = time.perf_counter() - started
        self.inference_executions = int(tracker.totals.get("executions", 0))
        logits -= logits.max(axis=1, keepdims=True)
        scores = np.exp(logits)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)
