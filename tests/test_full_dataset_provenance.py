import json
from types import SimpleNamespace

import experiments.full_dataset as full_dataset
from experiments.full_dataset import generation_provenance


def test_future_dataset_manifest_provenance_identifies_generator_sources_and_runtime():
    provenance = generation_provenance()

    assert provenance["generator"] == "experiments/full_dataset.py"
    assert "experiments/full_dataset.py" in provenance["source_sha256"]
    assert "experiments/run_experiment.py" in provenance["source_sha256"]
    assert {
        "src/quantum_simulator/statistics.py",
        "src/quantum_simulator/fault_localizer.py",
        "src/quantum_simulator/hardware_fault_localizer.py",
        "quantum_simulator.py",
        "diagnosis_engine.py",
        "hardware_model.py",
    } <= set(provenance["source_sha256"])
    assert all(len(value) == 64 for value in provenance["source_sha256"].values())
    assert provenance["runtime"]["python_version"]
    assert provenance["runtime"]["numpy_version"]


def test_generator_persists_source_and_runtime_provenance_without_simulating(tmp_path, monkeypatch):
    monkeypatch.setattr(full_dataset, "FAMILY_COUNT", 1)
    monkeypatch.setattr(
        full_dataset,
        "run_experiment",
        lambda config, *, output_path: SimpleNamespace(status="COMPLETED", errors=[]),
    )

    manifest = full_dataset.generate_dataset(tmp_path)
    saved = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert manifest["generation_provenance"] == saved["generation_provenance"]
    assert saved["generation_provenance"]["source_sha256"] == generation_provenance()["source_sha256"]
    assert saved["families"][0]["circuit_lineage_id"] == "design_recipe_01"
    assert all(config.circuit_lineage_id == "design_recipe_01" for config in full_dataset.build_family_configs(0))
