from experiments.pilot import PILOT_FAMILIES, REPETITIONS
from experiments.run_pilot import build_pilot_config


def test_pilot_metadata_groups_repetitions_and_assigns_truth_from_condition():
    configs = [
        build_pilot_config(family_id, label, repetition)
        for family_id in PILOT_FAMILIES
        for label in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
        for repetition in range(REPETITIONS)
    ]

    assert len(configs) == 45
    assert len({config.experimental_group_id for config in configs}) == 15
    assert len({config.run_id for config in configs}) == 45
    for family_id in PILOT_FAMILIES:
        family_configs = [config for config in configs if config.circuit_family_id == family_id]
        for label in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"):
            group = [config for config in family_configs if config.ground_truth_label == label]
            assert len(group) == 3
            assert len({config.experimental_group_id for config in group}) == 1
            assert sorted(config.repetition_index for config in group) == [0, 1, 2]
            assert len({config.run_id for config in group}) == 3
        assert all(config.observed_counts is None for config in family_configs)


def test_pilot_condition_metadata_is_explicit_and_applicability_is_enforced():
    normal = build_pilot_config("single_x", "NORMAL", 0)
    program = build_pilot_config("single_x", "PROGRAM_FAULT", 0)
    hardware = build_pilot_config("single_x", "HARDWARE_ANOMALY", 0)

    assert normal.program_fault_id is normal.hardware_condition_id is None
    assert program.program_fault_id and program.hardware_condition_id is None
    assert hardware.hardware_condition_id and hardware.program_fault_id is None
    assert len({normal.circuit_id, program.circuit_id, hardware.circuit_id}) == 1
    assert normal.experimental_group_id != program.experimental_group_id != hardware.experimental_group_id
