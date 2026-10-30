from qmutbench_qml.training.run_full_streaming import (
    _checkpointed_pause_fields,
    _resume_source_fingerprints,
    bounded_chunk_batches,
    should_pause_for_memory,
)
import pytest


def test_interrupted_stream_pause_cursor_uses_last_durable_checkpoint():
    progress = {"status": "running", "active_fold": 1, "active_chunk": 156}
    latest = {"chunk_index": 155, "epoch": 0, "next_batch": 2480}

    paused = _checkpointed_pause_fields(progress, 1, latest)

    assert paused["status"] == "paused_checkpointed"
    assert paused["active_chunk"] == 155
    assert paused["next_chunk"] == 156
    assert paused["next_epoch"] == 0
    assert paused["next_batch"] == 2480
    assert paused["last_checkpoint_validated"] is True


def test_full_streaming_pauses_at_durable_boundary_with_memory_reserve():
    assert should_pause_for_memory(1279)
    assert not should_pause_for_memory(1280)


def test_chunk_limit_shortens_at_epoch_boundary_without_crossing_it():
    assert bounded_chunk_batches(0, 14482, 16) == 16
    assert bounded_chunk_batches(14472, 14482, 16) == 10
    assert bounded_chunk_batches(14481, 14482, 16) == 1
    with pytest.raises(ValueError, match="Invalid epoch cursor"):
        bounded_chunk_batches(14482, 14482, 16)


def test_resume_preserves_checkpoint_identity_only_for_trainer_control_edits():
    stored = {"full_trainer_sha256": "old", "qnn_model_sha256": "model",
              "feature_loader_sha256": "loader"}
    current = {"full_trainer_sha256": "new", "qnn_model_sha256": "model",
               "feature_loader_sha256": "loader"}
    assert _resume_source_fingerprints(current, stored)["full_trainer_sha256"] == "old"

    current["qnn_model_sha256"] = "changed-model"
    with pytest.raises(ValueError, match="training-relevant source fingerprints changed"):
        _resume_source_fingerprints(current, stored)
