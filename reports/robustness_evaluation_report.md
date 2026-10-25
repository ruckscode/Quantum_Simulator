# Deterministic Pipeline Robustness Evaluation

This report evaluates the executable deterministic diagnosis/localization pipeline via `diagnose_execution`. No ML model or classifier is called; results are not ML accuracy.

## Exact evaluation design

- Cases: **1188**; shots per distribution: **128**; ideal reference samples are reused for repeated fault/noise variants with identical circuit and seed.
- Seeds: **11, 29, 47**.
- Unseen generated circuit families: GHZ, parity-network, and QFT-like; each is generated from requested width, depth, and seed.
- Widths: small=2 qubits, medium=3 qubits, larger=4 qubits; depths: 2, 4.
- Program faults: gate replacement, deletion, insertion, parameter modification, and explicit single-qubit and two-qubit gate faults. Faulty circuits are localized by the existing operation-sequence evidence. All affected qubits are retained as ground truth.
- Hardware scenarios: readout error, gate error, T1 degradation, T2 degradation, and combined degradation; strengths are 0.15, 0.45, and 0.8.
- For each seed/family/size/depth combination: one normal, six program fault, and fifteen hardware anomaly cases.

## Actual aggregate results

- Anomaly detection: **908/1188** (76.4%).
- PROGRAM_FAULT/HARDWARE_ANOMALY/NO_ANOMALY routing: **908/1188** (76.4%).
- Program fault localization (exact operation and qubits): **324/324** (100.0%).
- Hardware localization (affected component present): **810/810** (100.0%).
- End-to-end correct diagnosis and localization: **908/1188** (76.4%).

## Results by scenario type

| Type | Cases | Anomaly correct | Routing correct | Localization correct |
|---|---:|---:|---:|---:|
| combined | 162 | 158/162 | 158/162 | 162/162 |
| gate_deletion | 54 | 33/54 | 33/54 | 54/54 |
| gate_error | 162 | 150/162 | 150/162 | 162/162 |
| gate_insertion | 54 | 42/54 | 42/54 | 54/54 |
| gate_replacement | 54 | 34/54 | 34/54 | 54/54 |
| normal | 54 | 54/54 | 54/54 | n/a |
| parameter_modification | 54 | 18/54 | 18/54 | 54/54 |
| readout_error | 162 | 134/162 | 134/162 | 162/162 |
| single_qubit_fault | 54 | 49/54 | 49/54 | 54/54 |
| t1_degradation | 162 | 99/162 | 99/162 | 162/162 |
| t2_degradation | 162 | 105/162 | 105/162 | 162/162 |
| two_qubit_fault | 54 | 32/54 | 32/54 | 54/54 |

## Results by size and depth

| Width | Depth | Cases | Anomaly correct | Routing correct | End-to-end correct |
|---|---:|---:|---:|---:|---:|
| small (2 qubits) | 2 | 198 | 141/198 | 141/198 | 141/198 |
| small (2 qubits) | 4 | 198 | 155/198 | 155/198 | 155/198 |
| medium (3 qubits) | 2 | 198 | 147/198 | 147/198 | 147/198 |
| medium (3 qubits) | 4 | 198 | 157/198 | 157/198 | 157/198 |
| larger (4 qubits) | 2 | 198 | 148/198 | 148/198 | 148/198 |
| larger (4 qubits) | 4 | 198 | 160/198 | 160/198 | 160/198 |

## Failure cases

Total failures against combined expected routing/localization: **280**.

| Scenario | Fault | Hardware | Size | Depth | Strength | Predicted category | Failures |
|---|---|---|---|---:|---:|---|---:|
| HARDWARE_ANOMALY | none | combined | medium | 2 | 0.15 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | combined | small | 2 | 0.15 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | combined | small | 4 | 0.15 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | larger | 2 | 0.15 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | gate_error | larger | 2 | 0.45 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | larger | 2 | 0.8 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | medium | 2 | 0.15 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | medium | 2 | 0.45 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | medium | 2 | 0.8 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | small | 2 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | gate_error | small | 2 | 0.45 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | gate_error | small | 2 | 0.8 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | readout_error | larger | 4 | 0.15 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | readout_error | larger | 4 | 0.45 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | readout_error | larger | 4 | 0.8 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | readout_error | medium | 2 | 0.15 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | readout_error | medium | 4 | 0.15 | NO_ANOMALY | 5 |
| HARDWARE_ANOMALY | none | readout_error | medium | 4 | 0.45 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | readout_error | medium | 4 | 0.8 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | readout_error | small | 2 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | readout_error | small | 2 | 0.45 | NO_ANOMALY | 1 |
| HARDWARE_ANOMALY | none | readout_error | small | 4 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | readout_error | small | 4 | 0.45 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | readout_error | small | 4 | 0.8 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 4 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 4 | 0.45 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | larger | 4 | 0.8 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 4 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 4 | 0.45 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | medium | 4 | 0.8 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 4 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 4 | 0.45 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t1_degradation | small | 4 | 0.8 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 4 | 0.15 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 4 | 0.45 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t2_degradation | larger | 4 | 0.8 | NO_ANOMALY | 3 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 4 | 0.15 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 4 | 0.45 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t2_degradation | medium | 4 | 0.8 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 2 | 0.15 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 2 | 0.45 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 2 | 0.8 | NO_ANOMALY | 4 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 4 | 0.15 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 4 | 0.45 | NO_ANOMALY | 2 |
| HARDWARE_ANOMALY | none | t2_degradation | small | 4 | 0.8 | NO_ANOMALY | 2 |
| PROGRAM_FAULT | gate_deletion | healthy | larger | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_deletion | healthy | larger | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_deletion | healthy | medium | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_deletion | healthy | medium | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_deletion | healthy | small | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_deletion | healthy | small | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_insertion | healthy | larger | 2 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_insertion | healthy | medium | 2 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_insertion | healthy | small | 2 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_insertion | healthy | small | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_replacement | healthy | larger | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_replacement | healthy | larger | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_replacement | healthy | medium | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_replacement | healthy | medium | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | gate_replacement | healthy | small | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | gate_replacement | healthy | small | 4 | 0.0 | NO_ANOMALY | 2 |
| PROGRAM_FAULT | parameter_modification | healthy | larger | 2 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | parameter_modification | healthy | larger | 4 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | parameter_modification | healthy | medium | 2 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | parameter_modification | healthy | medium | 4 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | parameter_modification | healthy | small | 2 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | parameter_modification | healthy | small | 4 | 0.0 | NO_ANOMALY | 6 |
| PROGRAM_FAULT | single_qubit_fault | healthy | larger | 2 | 0.0 | NO_ANOMALY | 1 |
| PROGRAM_FAULT | single_qubit_fault | healthy | medium | 2 | 0.0 | NO_ANOMALY | 1 |
| PROGRAM_FAULT | single_qubit_fault | healthy | medium | 4 | 0.0 | NO_ANOMALY | 1 |
| PROGRAM_FAULT | single_qubit_fault | healthy | small | 2 | 0.0 | NO_ANOMALY | 1 |
| PROGRAM_FAULT | single_qubit_fault | healthy | small | 4 | 0.0 | NO_ANOMALY | 1 |
| PROGRAM_FAULT | two_qubit_fault | healthy | larger | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | two_qubit_fault | healthy | larger | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | two_qubit_fault | healthy | medium | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | two_qubit_fault | healthy | medium | 4 | 0.0 | NO_ANOMALY | 3 |
| PROGRAM_FAULT | two_qubit_fault | healthy | small | 2 | 0.0 | NO_ANOMALY | 4 |
| PROGRAM_FAULT | two_qubit_fault | healthy | small | 4 | 0.0 | NO_ANOMALY | 4 |

First 20 individual failure records (seed, truth, predictions and candidate evidence) are included below:

```json
[
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_replacement",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.0,
    "truth": {
      "operation": "rx",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Gate operation differs from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rx on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rx on qubits (0,).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_deletion",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Expected operation missing in observed circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "parameter_modification",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Parameter values differ from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "readout_error",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.45,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.45,
      "kind": "readout_error"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 1.8,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.4500"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 1.8,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.4500"
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.15,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.15,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.45,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.45,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 11,
    "strength": 0.8,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.8,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.8,
        "reason": "Qubit 0 shows hardware degradation: t2=20.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.8,
        "reason": "Qubit 0 shows hardware degradation: t2=20.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_replacement",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.0,
    "truth": {
      "operation": "rx",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Gate operation differs from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rx on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rx on qubits (0,).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_deletion",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Expected operation missing in observed circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "parameter_modification",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Parameter values differ from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "readout_error",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.15,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.15,
      "kind": "readout_error"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.6,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.1500"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.6,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.1500"
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.15,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.15,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.45,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.45,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 29,
    "strength": 0.8,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.8,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.8,
        "reason": "Qubit 0 shows hardware degradation: t2=20.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.8,
        "reason": "Qubit 0 shows hardware degradation: t2=20.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_replacement",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.0,
    "truth": {
      "operation": "rx",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Gate operation differs from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rx",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rx on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rx on qubits (0,).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "gate_deletion",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Expected operation missing in observed circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "PROGRAM_FAULT",
    "fault_type": "parameter_modification",
    "hardware_type": "healthy",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.0,
    "truth": {
      "operation": "rz",
      "qubits": [
        0
      ],
      "index": 2
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "PROGRAM_FAULT",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2,
        "reason": "Parameter values differ from expected circuit"
      }
    ],
    "program_evidence": [
      {
        "operation": "rz",
        "qubits": [
          0
        ],
        "index": 2
      }
    ],
    "hardware_candidates": [],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 2.0,
        "reason": "Program trace implicated qubit 0: No notable qubit degradation was detected."
      },
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 0.35,
        "reason": "Gate calibration for rz on qubits (0,): No gate calibration was available."
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "readout_error",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.15,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.15,
      "kind": "readout_error"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.6,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.1500"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.6,
        "reason": "Qubit 0 shows hardware degradation: readout_error=0.1500"
      }
    ],
    "missing_evidence": [
      "No gate calibration entry is available for h on qubits (0,).",
      "No gate calibration entry is available for cx on qubits (0, 1).",
      "No coupling-edge calibration is available for edge (0, 1).",
      "No gate calibration entry is available for rz on qubits (0,)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.15,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.15,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.15,
        "reason": "Qubit 0 shows hardware degradation: t2=85.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  },
  {
    "scenario": "HARDWARE_ANOMALY",
    "fault_type": "none",
    "hardware_type": "t2_degradation",
    "family": "ghz",
    "size": "small",
    "n_qubits": 2,
    "depth": 2,
    "seed": 47,
    "strength": 0.45,
    "truth": {
      "component_type": "qubit",
      "component": 0,
      "strength": 0.45,
      "kind": "t2_degradation"
    },
    "anomaly_truth": true,
    "anomaly_predicted": false,
    "category_truth": "HARDWARE_ANOMALY",
    "category_predicted": "NO_ANOMALY",
    "program_candidates": [
      {
        "operation": "measure",
        "qubits": [],
        "index": 5,
        "reason": "Largest traced deviation without direct operation mismatch"
      }
    ],
    "program_evidence": [],
    "hardware_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "hardware_result_candidates": [
      {
        "component_type": "gate",
        "component": [
          0
        ],
        "score": 10.0,
        "reason": "Gate calibration for rz on qubits (0,) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "gate",
        "component": [
          0,
          1
        ],
        "score": 10.0,
        "reason": "Gate calibration for cx on qubits (0, 1) is degraded: gate_duration=100.0000"
      },
      {
        "component_type": "qubit",
        "component": 0,
        "score": 0.44999999999999996,
        "reason": "Qubit 0 shows hardware degradation: t2=55.00"
      }
    ],
    "missing_evidence": [
      "No coupling-edge calibration is available for edge (0, 1)."
    ]
  }
]
```

## Localization performance

Program success requires the injected operation and exact affected qubit tuple to appear in positive program-fault evidence. Hardware success requires the affected component to appear among relevant hardware evidence candidates. These are top-candidate-inclusion measures, not rank or causal certainty measures.

## Limitations

- All observations are generated by the project statevector simulator and its HardwareModel; these are simulated, controlled anomaly conditions.
- Seed variability is measured with three seeds and 128 shots per distribution, across three generated circuit families, three widths, and two depths; this does not establish broad real-circuit generalization.
- Program and hardware scenarios are run as independent axes: hardware degradation is applied only in hardware-anomaly cases; program-fault cases use ideal hardware.
- Hardware T1/T2 values and strengths are controlled calibration inputs. Their simulator effects depend on operation duration and implemented noise semantics.
- No real ML model, external benchmark, or physical quantum hardware is used.
