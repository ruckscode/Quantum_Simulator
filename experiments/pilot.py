"""Fixed, controlled five-family pilot design for execution-derived data."""

PILOT_FAMILIES = {
    "single_x": {
        "reference": [["x", 0], ["measure", 0]],
        "mutated": [["h", 0], ["measure", 0]],
        "program_fault_id": "single_x_replace_x_with_h_v1",
        "program_fault_description": "Replace operation 0: X(q0) with H(q0).",
        "program_fault_operation_index": 0,
        "program_fault_qubits": [0],
    },
    "single_ry": {
        "reference": [["ry", 0, 1.0471975512], ["measure", 0]],
        "mutated": [["x", 0], ["measure", 0]],
        "program_fault_id": "single_ry_replace_ry_with_x_v1",
        "program_fault_description": "Replace operation 0: RY(q0, 1.0471975512) with X(q0).",
        "program_fault_operation_index": 0,
        "program_fault_qubits": [0],
    },
    "hzh_sequence": {
        "reference": [["h", 0], ["z", 0], ["h", 0], ["measure", 0]],
        "mutated": [["h", 0], ["x", 0], ["h", 0], ["measure", 0]],
        "program_fault_id": "hzh_replace_z_with_x_v1",
        "program_fault_description": "Replace operation 1: Z(q0) with X(q0).",
        "program_fault_operation_index": 1,
        "program_fault_qubits": [0],
    },
    "bell_pair": {
        "reference": [["h", 0], ["cx", 0, 1], ["measure"]],
        "mutated": [["h", 0], ["measure"]],
        "program_fault_id": "bell_pair_remove_cx_v1",
        "program_fault_description": "Delete operation 1: CX(q0,q1).",
        "program_fault_operation_index": 1,
        "program_fault_qubits": [0, 1],
    },
    "controlled_pair": {
        "reference": [["x", 0], ["h", 1], ["cx", 0, 1], ["measure"]],
        "mutated": [["h", 1], ["cx", 0, 1], ["measure"]],
        "program_fault_id": "controlled_pair_remove_x_v1",
        "program_fault_description": "Delete operation 0: X(q0).",
        "program_fault_operation_index": 0,
        "program_fault_qubits": [0],
    },
}

HARDWARE_READOUT_ERROR = {
    "single_x": 0.20,
    "single_ry": 0.25,
    "hzh_sequence": 0.30,
    "bell_pair": 0.35,
    "controlled_pair": 0.40,
}
REPETITIONS = 3
SHOTS = 2048
BASE_SEED = 73000
