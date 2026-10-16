import pytest

from quantum_simulator.hardware_model import HardwareModel


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("property_name", ["t1", "t2"])
def test_hardware_time_constants_reject_non_finite_values(property_name, bad_value):
    with pytest.raises(ValueError, match="finite"):
        HardwareModel(1, **{f"default_{property_name}": bad_value})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_hardware_error_probabilities_reject_non_finite_values(bad_value):
    with pytest.raises(ValueError, match="finite"):
        HardwareModel(1, default_readout_error=bad_value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_gate_calibration_rejects_non_finite_values(bad_value):
    hardware = HardwareModel(1)
    with pytest.raises(ValueError, match="finite"):
        hardware.add_gate_calibration("x", error=bad_value)
