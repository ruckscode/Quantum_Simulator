"""Regression tests for QNN epoch-only resume detection."""
from pathlib import Path

from qmutbench_qml.training import run_binary_development as binary


def test_resume_gate_accepts_epoch_only_checkpoint_candidates(tmp_path: Path):
    assert not binary.has_resume_candidates(tmp_path)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    assert not binary.has_resume_candidates(tmp_path)

    checkpoint_dir = tmp_path / "fold_checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fold_01_qnn_epoch_0001.json").write_text("{}", encoding="utf-8")
    assert binary.has_resume_candidates(tmp_path)


def test_resume_gate_rejects_out_of_range_fold_checkpoint_names(tmp_path: Path):
    checkpoint_dir = tmp_path / "fold_checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fold_06_qnn_epoch_0001.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "fold_00.json").write_text("{}", encoding="utf-8")
    assert not binary.has_resume_candidates(tmp_path)


def test_resume_gate_accepts_completed_fold_checkpoint_candidate(tmp_path: Path):
    checkpoint_dir = tmp_path / "fold_checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "fold_02.json").write_text("{}", encoding="utf-8")
    assert binary.has_resume_candidates(tmp_path)
