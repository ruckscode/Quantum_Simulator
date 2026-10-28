"""Analytic, trainable binary VQC for the QMutBench killed/survived target."""
from __future__ import annotations

import time

import pennylane as qml
import psutil
from pennylane import numpy as pnp

RAM_FLOOR_MIB = 1024


class BinaryVQC:
    """Angle re-uploading VQC with a trainable sigmoid readout."""

    def __init__(self, n_qubits=4, depth=2, epochs=8, learning_rate=.02,
                 batch_size=32, seed=20261028, optimizer="adam",
                 encoding="angle_reupload", device_name="default.qubit",
                 diff_method="backprop"):
        if n_qubits < 2 or depth < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("qubits>=2, depth>=1, epochs>=1, and batch_size>=1 are required")
        if optimizer not in {"adam", "gd"}:
            raise ValueError("optimizer must be adam or gd")
        if encoding != "angle_reupload":
            raise ValueError("Only angle_reupload is implemented and validated")
        self.n_qubits, self.depth, self.epochs = n_qubits, depth, epochs
        self.learning_rate, self.batch_size, self.seed = learning_rate, batch_size, seed
        self.optimizer_name, self.encoding = optimizer, encoding
        self.device_name, self.diff_method = device_name, diff_method
        self.device = qml.device(device_name, wires=n_qubits, shots=None, seed=seed)

        @qml.qnode(self.device, interface="autograd", diff_method=diff_method)
        def circuit(features, weights):
            for layer in range(depth):
                qml.AngleEmbedding(pnp.pi * features, wires=range(n_qubits), rotation="Y")
                for wire in range(n_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def _logits(self, x, weights, scale, bias):
        return pnp.stack([scale * self.circuit(row, weights) + bias for row in x])

    @staticmethod
    def _weighted_loss(logits, labels, weights):
        probabilities = 1.0 / (1.0 + pnp.exp(-logits))
        clipped = pnp.clip(probabilities, 1e-9, 1 - 1e-9)
        loss = -(labels * pnp.log(clipped) + (1 - labels) * pnp.log(1 - clipped))
        return pnp.mean(weights * loss)

    def gradient_check(self, x, y, epsilon=1e-5, tolerance=5e-4):
        """Compare one backprop derivative with a centered finite difference."""
        import numpy as np
        from autograd import grad

        features = pnp.array(x[0], requires_grad=False)
        label = int(y[0])
        weights = pnp.array(np.random.default_rng(self.seed).normal(
            0, .12, (self.depth, self.n_qubits, 3)), requires_grad=True)
        scale = pnp.array(1.0, requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)

        def loss_with_weights(candidate):
            logits = scale * self.circuit(features, candidate) + bias
            prob = 1.0 / (1.0 + pnp.exp(-logits))
            prob = pnp.clip(prob, 1e-9, 1 - 1e-9)
            return -(label * pnp.log(prob) + (1 - label) * pnp.log(1 - prob))

        gradient = np.asarray(grad(loss_with_weights)(weights))
        raw = np.asarray(weights).copy()
        theta = float(raw[0, 0, 0])
        plus, minus = raw.copy(), raw.copy()
        plus[0, 0, 0] += epsilon
        minus[0, 0, 0] -= epsilon
        autodiff = float(gradient[0, 0, 0])
        finite_difference = float((loss_with_weights(pnp.array(plus, requires_grad=False)) -
                                   loss_with_weights(pnp.array(minus, requires_grad=False))) / (2 * epsilon))
        error = abs(autodiff - finite_difference)
        result = {"passed": bool(error <= tolerance), "autodiff": autodiff,
                  "finite_difference": finite_difference, "absolute_error": error,
                  "tolerance": tolerance, "epsilon": epsilon}
        if not result["passed"]:
            raise ArithmeticError(f"QNN gradient check failed: {result}")
        return result

    @staticmethod
    def _plain(value):
        """Convert optimizer/RNG state into JSON-safe Python values."""
        if hasattr(value, "tolist"):
            return BinaryVQC._plain(value.tolist())
        if isinstance(value, dict):
            return {str(key): BinaryVQC._plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [BinaryVQC._plain(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    def fit(self, x, y, resume_state=None, checkpoint_callback=None):
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2 or x.shape[1] != self.n_qubits or len(x) != len(y):
            raise ValueError("Input shape must be (rows, n_qubits), aligned with labels")
        if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
            raise ValueError("Training requires both binary classes and finite features")
        if x.min() < -1e-9 or x.max() > 1 + 1e-9:
            raise ValueError("Angle inputs must be fold-scaled to [0,1]")
        self.gradient_check_result = self.gradient_check(x[:1], y[:1])

        counts = np.bincount(y, minlength=2)
        class_weights = len(y) / (2.0 * counts)
        row_weights = class_weights[y]
        rng = np.random.default_rng(self.seed)
        if resume_state is None:
            weights = pnp.array(rng.normal(0, .12, (self.depth, self.n_qubits, 3)), requires_grad=True)
            scale = pnp.array(1.0, requires_grad=True)
            bias = pnp.array(0.0, requires_grad=True)
            start_epoch = 0
            self.loss_history = []
            prior_training_seconds = 0.0
            prior_training_executions = 0
            prior_peak_rss = 0
        else:
            start_epoch = int(resume_state["completed_epochs"])
            if start_epoch > self.epochs or len(resume_state["loss_history"]) != start_epoch:
                raise ValueError("QNN epoch checkpoint is inconsistent with the configured epoch budget")
            weights = pnp.array(resume_state["weights"], requires_grad=True)
            scale = pnp.array(resume_state["readout_scale"], requires_grad=True)
            bias = pnp.array(resume_state["readout_bias"], requires_grad=True)
            rng.bit_generator.state = resume_state["rng_state"]
            self.loss_history = [float(value) for value in resume_state["loss_history"]]
            prior_training_seconds = float(resume_state["training_seconds"])
            prior_training_executions = int(resume_state["training_executions"])
            prior_peak_rss = int(resume_state["peak_rss_bytes"])
        optimizer = (qml.AdamOptimizer(stepsize=self.learning_rate) if self.optimizer_name == "adam"
                     else qml.GradientDescentOptimizer(stepsize=self.learning_rate))
        if resume_state is not None and resume_state["optimizer_accumulation"] is not None:
            accumulation = resume_state["optimizer_accumulation"]
            if not isinstance(optimizer, qml.AdamOptimizer):
                raise ValueError("Checkpoint contains Adam state for a non-Adam optimizer")
            optimizer.accumulation = {
                "fm": [pnp.array(item, requires_grad=False) if isinstance(item, list) else item
                       for item in accumulation["fm"]],
                "sm": [pnp.array(item, requires_grad=False) if isinstance(item, list) else item
                       for item in accumulation["sm"]],
                "t": int(accumulation["t"])}
        started = time.perf_counter()
        self.peak_rss_bytes = max(prior_peak_rss, psutil.Process().memory_info().rss)
        with qml.Tracker(self.device) as tracker:
            for epoch in range(start_epoch, self.epochs):
                if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                    raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB during QNN fit")
                order = rng.permutation(len(y))
                epoch_loss = 0.0
                observations = 0
                for start in range(0, len(y), self.batch_size):
                    if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                        raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB in epoch {epoch + 1}")
                    batch = order[start:start + self.batch_size]
                    batch_x = pnp.array(x[batch], requires_grad=False)
                    batch_y = pnp.array(y[batch], requires_grad=False)
                    batch_w = pnp.array(row_weights[batch], requires_grad=False)

                    def objective(param, readout_scale, readout_bias):
                        logits = self._logits(batch_x, param, readout_scale, readout_bias)
                        return self._weighted_loss(logits, batch_y, batch_w)

                    (weights, scale, bias), cost = optimizer.step_and_cost(
                        objective, weights, scale, bias)
                    self.peak_rss_bytes = max(self.peak_rss_bytes, psutil.Process().memory_info().rss)
                    epoch_loss += float(cost) * len(batch)
                    observations += len(batch)
                self.loss_history.append(epoch_loss / observations)
                if checkpoint_callback is not None:
                    accumulation = getattr(optimizer, "accumulation", None)
                    checkpoint_callback({
                        "completed_epochs": epoch + 1,
                        "weights": self._plain(weights),
                        "readout_scale": float(scale),
                        "readout_bias": float(bias),
                        "loss_history": list(self.loss_history),
                        "rng_state": self._plain(rng.bit_generator.state),
                        "optimizer_accumulation": self._plain(accumulation),
                        "training_seconds": prior_training_seconds + time.perf_counter() - started,
                        "training_executions": prior_training_executions +
                            int(tracker.totals.get("executions", 0)),
                        "peak_rss_bytes": int(self.peak_rss_bytes)})
                print(f"[binary-vqc] epoch {epoch + 1}/{self.epochs} "
                      f"loss={self.loss_history[-1]:.6f} RAM={psutil.virtual_memory().available / 2**20:.0f} MiB",
                      flush=True)
        self.weights, self.readout_scale, self.readout_bias = weights, scale, bias
        self.training_seconds = prior_training_seconds + time.perf_counter() - started
        self.training_executions = prior_training_executions + int(tracker.totals.get("executions", 0))
        return self

    def predict_proba(self, x):
        import numpy as np
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_qubits or not np.isfinite(x).all():
            raise ValueError("Prediction inputs must be finite with n_qubits columns")
        if len(x) and (x.min() < -1e-9 or x.max() > 1 + 1e-9):
            raise ValueError("Angle inputs must use the training fold's [0,1] transform")
        if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
            raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB before QNN inference")
        started = time.perf_counter()
        parts = []
        with qml.Tracker(self.device) as tracker:
            for start in range(0, len(x), self.batch_size):
                if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                    raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB during QNN inference")
                stop = min(start + self.batch_size, len(x))
                # Keep the quantum expectation separate from the trained affine
                # readout here; probabilities below apply scale and bias once.
                logits = self._logits(pnp.array(x[start:stop], requires_grad=False), self.weights,
                                      pnp.array(1.0), pnp.array(0.0))
                parts.append(np.asarray(logits, dtype=float))
        logits = np.concatenate(parts) if parts else np.empty(0, dtype=float)
        self.inference_seconds = time.perf_counter() - started
        self.inference_executions = int(tracker.totals.get("executions", 0))
        p1 = 1.0 / (1.0 + np.exp(-np.clip(float(self.readout_scale) * logits + float(self.readout_bias), -40, 40)))
        return p1

    def configuration(self):
        return {"architecture": "angle-reuploading Rot/CNOT binary VQC",
            "qubits": self.n_qubits, "depth": self.depth,
            "trainable_parameters": self.depth * self.n_qubits * 3 + 2,
            "encoding": self.encoding, "ansatz": "per-layer RY(pi*x), trainable Rot per wire, ring CNOT",
            "readout": "trainable affine transform of <Z0>, sigmoid probability",
            "optimizer": self.optimizer_name, "learning_rate": self.learning_rate,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "simulator": self.device_name, "shots": None, "diff_method": self.diff_method}
