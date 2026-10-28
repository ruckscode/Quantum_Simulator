"""Analytic, trainable binary VQC for the QMutBench killed/survived target."""
from __future__ import annotations

import time
import math
import hashlib

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

    def fit(self, x, y, resume_state=None, checkpoint_callback=None,
            max_batches_per_chunk=None, compact_batch_order=False):
        import numpy as np

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        if x.ndim != 2 or x.shape[1] != self.n_qubits or len(x) != len(y):
            raise ValueError("Input shape must be (rows, n_qubits), aligned with labels")
        if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
            raise ValueError("Training requires both binary classes and finite features")
        if x.min() < -1e-9 or x.max() > 1 + 1e-9:
            raise ValueError("Angle inputs must be fold-scaled to [0,1]")
        if resume_state is None:
            self.gradient_check_result = self.gradient_check(x[:1], y[:1])
        else:
            self.gradient_check_result = resume_state.get("gradient_check_result")
            if self.gradient_check_result is None:
                self.gradient_check_result = self.gradient_check(x[:1], y[:1])

        counts = np.bincount(y, minlength=2)
        class_weights = len(y) / (2.0 * counts)
        row_weights = class_weights[y]
        rng = np.random.default_rng(self.seed)
        if resume_state is None:
            weights = pnp.array(rng.normal(0, .12, (self.depth, self.n_qubits, 3)), requires_grad=True)
            initial_weights = self._plain(weights)
            scale = pnp.array(1.0, requires_grad=True)
            bias = pnp.array(0.0, requires_grad=True)
            start_epoch = 0
            self.loss_history = []
            prior_training_seconds = 0.0
            prior_training_executions = 0
            prior_peak_rss = 0
            epoch = 0
            order = None
            next_batch = 0
            epoch_loss_sum = 0.0
            epoch_observations = 0
            chunk_index = 0
            processed_records = 0
            chunk_start_records = 0
            chunk_loss_sum = 0.0
            chunk_loss_records = 0
            shuffle_rng_state = None
        else:
            start_epoch = int(resume_state["completed_epochs"])
            if start_epoch > self.epochs or len(resume_state["loss_history"]) != start_epoch:
                raise ValueError("QNN epoch checkpoint is inconsistent with the configured epoch budget")
            weights = pnp.array(resume_state["weights"], requires_grad=True)
            initial_weights = resume_state.get("initial_weights")
            if initial_weights is None:
                initial_weights = np.random.default_rng(self.seed).normal(
                    0, .12, (self.depth, self.n_qubits, 3)).tolist()
            scale = pnp.array(resume_state["readout_scale"], requires_grad=True)
            bias = pnp.array(resume_state["readout_bias"], requires_grad=True)
            rng.bit_generator.state = resume_state["rng_state"]
            self.loss_history = [float(value) for value in resume_state["loss_history"]]
            prior_training_seconds = float(resume_state["training_seconds"])
            prior_training_executions = int(resume_state["training_executions"])
            prior_peak_rss = int(resume_state["peak_rss_bytes"])
            epoch = int(resume_state.get("epoch", start_epoch))
            batch_order_state = resume_state.get("batch_order")
            compact_batch_order = bool(resume_state.get("compact_batch_order", compact_batch_order))
            shuffle_rng_state = resume_state.get("shuffle_rng_state")
            if isinstance(batch_order_state, dict):
                if batch_order_state.get("algorithm") != "numpy-pcg64-permutation-v1" or not shuffle_rng_state:
                    raise ValueError("Compact batch order checkpoint is invalid")
                order_rng = np.random.default_rng()
                order_rng.bit_generator.state = shuffle_rng_state
                order = order_rng.permutation(len(y))
                order_hash = hashlib.sha256(order.astype(np.int64, copy=False).tobytes()).hexdigest()
                if (len(order) != int(batch_order_state.get("length", -1)) or
                        order_hash != batch_order_state.get("sha256")):
                    raise ValueError("Deterministic batch order does not match its checkpoint digest")
            else:
                order = None if batch_order_state is None else np.asarray(batch_order_state, dtype=np.int64)
            next_batch = int(resume_state.get("next_batch", 0))
            epoch_loss_sum = float(resume_state.get("epoch_loss_sum", 0.0))
            epoch_observations = int(resume_state.get("epoch_observations", 0))
            chunk_index = int(resume_state.get("chunk_index", 0))
            processed_records = int(resume_state.get("processed_records", 0))
            chunk_start_records = processed_records
            # Every resumable state is a committed chunk boundary; start fresh
            # chunk metrics while preserving partial-epoch cumulative loss.
            chunk_loss_sum = 0.0
            chunk_loss_records = 0
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
            batches_this_chunk = 0
            while epoch < self.epochs:
                if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                    raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB during QNN fit")
                if order is None:
                    shuffle_rng_state = self._plain(rng.bit_generator.state)
                    order = rng.permutation(len(y))
                    next_batch = 0
                    epoch_loss_sum = 0.0
                    epoch_observations = 0
                batch_count = math.ceil(len(y) / self.batch_size)
                for batch_index in range(next_batch, batch_count):
                    if psutil.virtual_memory().available / 2**20 < RAM_FLOOR_MIB:
                        raise MemoryError(f"Available RAM fell below {RAM_FLOOR_MIB} MiB in epoch {epoch + 1}")
                    start = batch_index * self.batch_size
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
                    epoch_loss_sum += float(cost) * len(batch)
                    epoch_observations += len(batch)
                    chunk_loss_sum += float(cost) * len(batch)
                    chunk_loss_records += len(batch)
                    processed_records += len(batch)
                    del batch_x, batch_y, batch_w, objective
                    next_batch = batch_index + 1
                    batches_this_chunk += 1
                    if next_batch == batch_count:
                        self.loss_history.append(epoch_loss_sum / epoch_observations)
                        epoch += 1
                        order = None
                        next_batch = 0
                        epoch_loss_sum = 0.0
                        epoch_observations = 0
                    should_pause = (max_batches_per_chunk is not None and
                                    batches_this_chunk >= max_batches_per_chunk)
                    checkpoint_due = (max_batches_per_chunk is None or should_pause or
                                      epoch >= self.epochs)
                    if checkpoint_callback is not None and checkpoint_due:
                        accumulation = getattr(optimizer, "accumulation", None)
                        serialized_order = None if order is None else (
                            self._order_reference(order) if compact_batch_order else order.tolist())
                        state = {
                        "completed_epochs": epoch,
                        "epoch": epoch,
                        "chunk_index": chunk_index + 1,
                        "next_batch": next_batch,
                        "batch_order": serialized_order,
                        "compact_batch_order": bool(compact_batch_order),
                        "shuffle_rng_state": shuffle_rng_state,
                        "batch_order_sha256": None if order is None else hashlib.sha256(
                            order.astype(np.int64, copy=False).tobytes()).hexdigest(),
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_observations": epoch_observations,
                        "processed_records": processed_records,
                        "chunk_records": processed_records - chunk_start_records,
                        "chunk_loss_sum": chunk_loss_sum,
                        "chunk_loss_records": chunk_loss_records,
                        "chunk_loss_mean": chunk_loss_sum / chunk_loss_records,
                        "gradient_check_result": self.gradient_check_result,
                        "initial_weights": initial_weights,
                        "parameter_update_verified": not np.allclose(
                            np.asarray(weights), np.asarray(initial_weights)),
                        "weights": self._plain(weights),
                        "readout_scale": float(scale),
                        "readout_bias": float(bias),
                        "loss_history": list(self.loss_history),
                        "rng_state": self._plain(rng.bit_generator.state),
                        "optimizer_accumulation": self._plain(accumulation),
                        "training_seconds": prior_training_seconds + time.perf_counter() - started,
                        "training_executions": prior_training_executions +
                            int(tracker.totals.get("executions", 0)),
                        "peak_rss_bytes": int(self.peak_rss_bytes),
                        "available_ram_bytes": int(psutil.virtual_memory().available)}
                        checkpoint_callback(state)
                        chunk_index += 1
                        chunk_start_records = processed_records
                        chunk_loss_sum = 0.0
                        chunk_loss_records = 0
                    if checkpoint_due:
                        batches_this_chunk = 0
                    if should_pause:
                        self.resume_state = self._resume_payload(
                            epoch, chunk_index, next_batch, order, epoch_loss_sum,
                            epoch_observations, processed_records, weights, scale, bias,
                            rng, optimizer, prior_training_seconds + time.perf_counter() - started,
                            prior_training_executions + int(tracker.totals.get("executions", 0)),
                            self.peak_rss_bytes, self.loss_history, compact_batch_order,
                            shuffle_rng_state, initial_weights)
                        self.training_seconds = self.resume_state["training_seconds"]
                        self.training_executions = self.resume_state["training_executions"]
                        self.training_complete = epoch >= self.epochs
                        self.weights, self.readout_scale, self.readout_bias = weights, scale, bias
                        return self
                print(f"[binary-vqc] epoch {epoch}/{self.epochs} "
                      f"RAM={psutil.virtual_memory().available / 2**20:.0f} MiB", flush=True)
        self.weights, self.readout_scale, self.readout_bias = weights, scale, bias
        self.training_seconds = prior_training_seconds + time.perf_counter() - started
        self.training_executions = prior_training_executions + int(tracker.totals.get("executions", 0))
        self.training_complete = True
        self.resume_state = self._resume_payload(
            epoch, chunk_index, next_batch, order, epoch_loss_sum, epoch_observations,
            processed_records, weights, scale, bias, rng, optimizer,
            self.training_seconds, self.training_executions, self.peak_rss_bytes,
            self.loss_history, compact_batch_order, shuffle_rng_state, initial_weights)
        return self

    def _resume_payload(self, epoch, chunk_index, next_batch, order, epoch_loss_sum,
                        epoch_observations, processed_records, weights, scale, bias,
                        rng, optimizer, prior_seconds, prior_executions, peak_rss,
                        loss_history, compact_batch_order=False, shuffle_rng_state=None,
                        initial_weights=None):
        import numpy as np
        order_state = None if order is None else (
            self._order_reference(order) if compact_batch_order else self._plain(order))
        return {"completed_epochs": int(epoch), "epoch": int(epoch),
            "chunk_index": int(chunk_index), "next_batch": int(next_batch),
            "batch_order": order_state, "compact_batch_order": bool(compact_batch_order),
            "shuffle_rng_state": shuffle_rng_state,
            "batch_order_sha256": None if order is None else hashlib.sha256(
                order.astype("int64", copy=False).tobytes()).hexdigest(),
            "epoch_loss_sum": float(epoch_loss_sum),
            "epoch_observations": int(epoch_observations),
            "processed_records": int(processed_records),
            "weights": self._plain(weights), "initial_weights": initial_weights,
            "parameter_update_verified": not np.allclose(
                np.asarray(weights), np.asarray(initial_weights)),
            "readout_scale": float(scale),
            "readout_bias": float(bias), "loss_history": list(loss_history),
            "rng_state": self._plain(rng.bit_generator.state),
            "optimizer_accumulation": self._plain(getattr(optimizer, "accumulation", None)),
            "training_seconds": float(prior_seconds),
            "training_executions": int(prior_executions),
            "peak_rss_bytes": int(peak_rss),
            "gradient_check_result": self.gradient_check_result}

    @staticmethod
    def _order_reference(order):
        import numpy as np
        return {"algorithm": "numpy-pcg64-permutation-v1", "length": int(len(order)),
                "sha256": hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()}

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
