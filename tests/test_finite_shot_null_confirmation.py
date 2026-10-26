import base64
import hashlib
import json
import subprocess
import sys
import tracemalloc
import uuid
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import finite_shot_null_confirmation as confirm
from experiments import finite_shot_synthetic_characterization as original_runner


def registered_inputs():
    prereg = json.loads(confirm.PREREG_PATH.read_text(encoding="utf-8"))
    binding = json.loads(confirm.BINDING_PATH.read_text(encoding="utf-8"))
    return prereg, binding


def make_approval(tmp_path, *, prereg=None, binding_path=None, driver_path=None, prereg_path=None):
    prereg = prereg or registered_inputs()[0]
    binding_path = binding_path or confirm.BINDING_PATH
    driver_path = driver_path or Path(confirm.__file__)
    prereg_path = prereg_path or confirm.PREREG_PATH
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    reviewer_id = "test-reviewer-key-01"
    key_id = hashlib.sha256(public).hexdigest()
    trust_root_id = "test-only-trust-root-20261016"
    trusted_path = tmp_path / "trusted_reviewer.json"
    trusted_path.write_text(json.dumps({
        "schema": "finite-shot-trusted-reviewer-v1",
        "reviewer_id": reviewer_id,
        "key_id": key_id,
        "trust_root_id": trust_root_id,
        "reviewer_independence_verified": True,
        "ed25519_public_key_hex": public.hex(),
    }), encoding="utf-8")
    approval = {
        "schema": "finite-shot-independent-approval-v1",
        "reviewer_id": reviewer_id,
        "key_id": key_id,
        "trust_root_id": trust_root_id,
        "run_id": str(uuid.uuid4()),
        "reviewed_at": "2030-01-01T00:00:00Z",
        "decision": "approved",
        "independence_basis": "test fixture only; not a real reviewer attestation",
        "review_scope": [
            "preregistration", "driver", "detector and statistical dependencies",
            "configuration", "seed ledger", "resource controls",
        ],
        "artifact_manifest": confirm.artifact_manifest(
            prereg, binding_path, driver_path, prereg_path, trust_root_id
        ),
    }
    approval["signature_base64"] = base64.b64encode(
        private.sign(confirm._canonical_bytes(approval))
    ).decode("ascii")
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    return trusted_path, approval_path


def test_registered_hash_binding_passes_or_fails_closed_on_detector_mismatch():
    prereg, binding = registered_inputs()
    confirm.validate_preregistration(prereg)
    registered_hash = prereg["detector"]["statistics_module_sha256"]
    current_hash = confirm.sha256_file(confirm.STATISTICS_MODULE)

    # Never rewrite a preregistered digest to fit the current source. If the
    # original detector is unavailable, the expected and current hashes must
    # remain visibly different and the confirmation binding must reject it.
    if current_hash != registered_hash:
        with pytest.raises(ValueError, match="statistics-module hash differs from preregistration"):
            confirm.verify_hash_bindings(prereg, binding)
    else:
        confirm.verify_hash_bindings(prereg, binding)
    assert binding["independent_review_release"]["status"] == "pending"


def test_modified_or_missing_approval_and_trust_key_fail_closed(tmp_path):
    prereg, _ = registered_inputs()
    trusted, approval = make_approval(tmp_path, prereg=prereg)
    confirm.verify_signed_approval(
        prereg, trusted_reviewer_path=trusted, approval_path=approval
    )
    approval.unlink()
    with pytest.raises(ValueError, match="inaccessible|missing/malformed"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=trusted, approval_path=approval
        )
    approval.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="missing/malformed"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=trusted, approval_path=approval
        )
    trusted.unlink()
    with pytest.raises(ValueError, match="inaccessible|missing/malformed"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=trusted, approval_path=approval
        )


def test_duplicate_json_keys_are_rejected_before_approval_interpretation(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"finite-shot-trusted-reviewer-v1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        confirm.strict_json_file(path)
    prereg, _ = registered_inputs()
    trusted, approval = make_approval(tmp_path, prereg=prereg)
    raw = approval.read_text(encoding="utf-8")
    approval.write_text(raw.replace("{", '{"schema":"shadow",', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=trusted, approval_path=approval
        )


def test_stale_malformed_and_wrong_identity_approval_fail_closed(tmp_path):
    prereg, _ = registered_inputs()
    trusted, approval_path = make_approval(tmp_path, prereg=prereg)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["independence_basis"] = "changed after signing"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="signature is invalid"):
        confirm.verify_signed_approval(prereg, trusted_reviewer_path=trusted, approval_path=approval_path)
    approval = json.loads((tmp_path / "approval.json").read_text(encoding="utf-8"))
    # Restore a valid signature from a fresh deterministic test fixture before exercising status checks.
    trusted, approval_path = make_approval(tmp_path, prereg=prereg)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["decision"] = "pending"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="pending or rejected"):
        confirm.verify_signed_approval(prereg, trusted_reviewer_path=trusted, approval_path=approval_path)
    approval["decision"] = "approved"
    approval["reviewer_id"] = "untrusted-identity"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        confirm.verify_signed_approval(prereg, trusted_reviewer_path=trusted, approval_path=approval_path)
    approval["reviewer_id"] = "test-reviewer-key-01"
    approval["signature_base64"] = "not-base64!"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="signature is invalid"):
        confirm.verify_signed_approval(prereg, trusted_reviewer_path=trusted, approval_path=approval_path)


def test_approval_is_bound_to_every_manifest_artifact(tmp_path, monkeypatch):
    prereg, _ = registered_inputs()
    driver = tmp_path / "driver.py"
    driver.write_bytes(Path(confirm.__file__).read_bytes())
    binding = tmp_path / "binding.json"
    binding.write_text(confirm.BINDING_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_bytes(confirm.PREREG_PATH.read_bytes())
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_bytes((confirm.ROOT / "pyproject.toml").read_bytes())
    trusted, approval_path = make_approval(
        tmp_path, prereg=prereg, binding_path=binding, driver_path=driver,
        prereg_path=prereg_path,
    )
    confirm.verify_signed_approval(
        prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
        trusted_reviewer_path=trusted, approval_path=approval_path,
    )
    original = driver.read_bytes()
    driver.write_bytes(original + b"\n# altered\n")
    with pytest.raises(ValueError, match="manifest is stale"):
        confirm.verify_signed_approval(
            prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )
    driver.write_bytes(original)

    altered_prereg = dict(prereg)
    altered_prereg["sampling"] = dict(prereg["sampling"], shot_count=2048)
    with pytest.raises(ValueError):
        confirm.verify_signed_approval(
            altered_prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )
    altered_ledger = json.loads(json.dumps(prereg))
    altered_ledger["sampling"]["seed_policy"]["seed_ledger_integrity"]["ordered_seed_bytes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seed ledger digest"):
        confirm.verify_signed_approval(
            altered_ledger, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )

    for module_attr in ("STATISTICS_MODULE", "REFERENCE_RUNNER"):
        source = getattr(confirm, module_attr)
        replacement = tmp_path / f"{module_attr}.py"
        replacement.write_bytes(source.read_bytes() + b"\n# altered\n")
        monkeypatch.setattr(confirm, module_attr, replacement)
        with pytest.raises(ValueError, match="manifest is stale"):
            confirm.verify_signed_approval(
                prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
                trusted_reviewer_path=trusted, approval_path=approval_path,
            )
        monkeypatch.setattr(confirm, module_attr, source)

    for path in (binding, prereg_path):
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="manifest is stale"):
            confirm.verify_signed_approval(
                prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
                trusted_reviewer_path=trusted, approval_path=approval_path,
            )
        path.write_bytes(original)

    monkeypatch.setattr(confirm, "ROOT", project_root)
    (project_root / "pyproject.toml").write_bytes((project_root / "pyproject.toml").read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="manifest is stale"):
        confirm.verify_signed_approval(
            prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )
    monkeypatch.undo()
    monkeypatch.setattr(confirm.cryptography, "__version__", "altered-crypto-version")
    with pytest.raises(ValueError, match="manifest is stale"):
        confirm.verify_signed_approval(
            prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )

    monkeypatch.undo()
    monkeypatch.setattr(confirm.np, "__version__", "altered-runtime-version")
    with pytest.raises(ValueError, match="manifest is stale"):
        confirm.verify_signed_approval(
            prereg, binding_path=binding, driver_path=driver, prereg_path=prereg_path,
            trusted_reviewer_path=trusted, approval_path=approval_path,
        )


def test_replacement_caller_hashes_are_not_an_authorization_input():
    assert "expected_driver_sha256" not in confirm.verify_hash_bindings.__annotations__
    assert "--expected-driver-sha256" not in confirm.main.__code__.co_consts
    with pytest.raises(TypeError):
        confirm.verify_hash_bindings({}, {}, expected_driver_sha256="f" * 64)


def test_pending_or_missing_review_never_reaches_sampling(tmp_path, monkeypatch):
    prereg, binding = registered_inputs()
    trusted, approval = make_approval(tmp_path, prereg=prereg)
    pending = json.loads(approval.read_text(encoding="utf-8"))
    pending["decision"] = "pending"
    approval.write_text(json.dumps(pending), encoding="utf-8")
    monkeypatch.setattr(confirm, "TRUSTED_REVIEWER_PATH", trusted)
    monkeypatch.setattr(confirm, "APPROVAL_PATH", approval)
    monkeypatch.setattr(confirm, "verify_hash_bindings", lambda *args, **kwargs: None)
    monkeypatch.setattr(confirm, "pair_outcome", lambda *args: pytest.fail("sampling was reached"))
    with pytest.raises(ValueError, match="pending or rejected"):
        confirm.execute_confirmation(execute_sampling=True)


def test_test_key_approval_is_only_accepted_against_separately_supplied_trust_key(tmp_path):
    prereg, _ = registered_inputs()
    trusted, approval = make_approval(tmp_path, prereg=prereg)
    confirm.verify_signed_approval(
        prereg, trusted_reviewer_path=trusted, approval_path=approval
    )
    wrong_trusted = tmp_path / "wrong_trusted.json"
    wrong_key = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    wrong_trusted.write_text(json.dumps({
        "schema": "finite-shot-trusted-reviewer-v1",
        "reviewer_id": "test-reviewer-key-01",
        "key_id": hashlib.sha256(wrong_key).hexdigest(),
        "trust_root_id": "test-only-trust-root-20261016",
        "reviewer_independence_verified": True,
        "ed25519_public_key_hex": wrong_key.hex(),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="key identifier does not match"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=wrong_trusted, approval_path=approval
        )


def test_consumed_run_id_rejects_approval_replay(tmp_path):
    prereg, _ = registered_inputs()
    trusted, approval_path = make_approval(tmp_path, prereg=prereg)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    run_id = approval["run_id"]
    assert confirm.verify_signed_approval(
        prereg, trusted_reviewer_path=trusted, approval_path=approval_path
    ) == run_id
    confirm.consume_approval_run(run_id, trust_root=tmp_path)
    with pytest.raises(ValueError, match="already consumed"):
        confirm.verify_signed_approval(
            prereg, trusted_reviewer_path=trusted, approval_path=approval_path
        )


def test_interrupted_preflight_consumes_authorization_and_cannot_resume(tmp_path, monkeypatch):
    prereg, _ = registered_inputs()
    trusted, approval_path = make_approval(tmp_path, prereg=prereg)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    run_id = approval["run_id"]
    monkeypatch.setattr(confirm, "TRUSTED_REVIEWER_PATH", trusted)
    monkeypatch.setattr(confirm, "APPROVAL_PATH", approval_path)
    monkeypatch.setattr(confirm, "TRUST_ROOT", tmp_path)
    # Isolate single-use authorization semantics from the separately tested
    # production hash gate; the registered detector digest is intentionally
    # mismatched in this checkout and must still block real execution.
    monkeypatch.setattr(confirm, "verify_hash_bindings", lambda *args, **kwargs: None)
    monkeypatch.setattr(confirm, "check_live_resources", lambda *args: (_ for _ in ()).throw(TimeoutError("simulated preflight stop")))
    monkeypatch.setattr(confirm.np.random, "default_rng", lambda *args: pytest.fail("sampling must not be reached"))
    try:
        with pytest.raises(TimeoutError, match="simulated preflight stop"):
            confirm.execute_confirmation(execute_sampling=True)
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    assert (tmp_path / f"{run_id}.consumed").is_file()
    with pytest.raises(ValueError, match="already consumed"):
        confirm.execute_confirmation(execute_sampling=True)


def test_symlinked_trust_record_is_rejected_when_platform_supports_symlinks(tmp_path):
    real = tmp_path / "real.json"
    link = tmp_path / "linked.json"
    real.write_text('{"ok":true}', encoding="utf-8")
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted by this Windows environment")
    with pytest.raises(ValueError, match="symlink or reparse point"):
        confirm.strict_json_file(link)


def test_junctioned_trust_directory_is_rejected_when_platform_supports_junctions(tmp_path):
    """Exercise Windows directory reparse points even when symlink privilege is absent."""
    if sys.platform != "win32":
        pytest.skip("Windows directory junction semantics are platform-specific")
    target = tmp_path / "junction-target"
    junction = tmp_path / "junction-link"
    target.mkdir()
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not junction.exists():
        pytest.skip(f"directory junction creation unavailable: {completed.stderr or completed.stdout}")
    try:
        with pytest.raises(ValueError, match="symlink or reparse point"):
            confirm._assert_directory_no_reparse(junction)
    finally:
        subprocess.run(
            ["cmd", "/c", "rmdir", str(junction)],
            capture_output=True,
            text=True,
            check=False,
        )


def test_shadowed_loaded_sampler_or_detector_is_rejected(monkeypatch):
    monkeypatch.setattr(confirm, "pair_outcome", lambda *args: False)
    with pytest.raises(ValueError, match="loaded reference runner"):
        confirm.verify_loaded_code_origins()


def test_execution_signature_has_no_trust_path_or_manifest_override():
    params = set(__import__("inspect").signature(confirm.execute_confirmation).parameters)
    assert not ({"trusted_reviewer_path", "approval_path", "expected_manifest", "expected_hash"} & params)
    with pytest.raises(TypeError):
        confirm.execute_confirmation(trusted_reviewer_path=Path("attacker-key.json"))


def test_public_pair_helper_reaches_rng_without_confirmation_authorization(monkeypatch):
    class ReachedRNG(Exception):
        pass

    monkeypatch.setattr(
        original_runner.np.random, "default_rng",
        lambda *args: (_ for _ in ()).throw(ReachedRNG("public helper reached RNG")),
    )
    with pytest.raises(ReachedRNG, match="public helper reached RNG"):
        original_runner.pair_outcome(
            1024, "null_equal_confirmation_20261016_01", 0,
            (0.5, 0.25, 0.15, 0.1), (0.5, 0.25, 0.15, 0.1),
        )


def test_exact_registered_workload_boundary_and_seed_ledger_without_sampling():
    prereg, _ = registered_inputs()
    confirm.validate_preregistration(prereg)
    assert prereg["sampling"]["replicates"] == 38228
    assert prereg["resource_cap"]["count_pairs_planned_and_maximum"] == 38228
    passing = confirm.decision_for_flags(2217)
    adjacent = confirm.decision_for_flags(2218)
    assert passing["verdict"] == "PASS"
    assert passing["one_sided_95_upper"] == pytest.approx(0.05999850882877437, abs=2e-12)
    assert adjacent["verdict"] == "INCONCLUSIVE"
    assert adjacent["one_sided_95_upper"] > 0.06
    first = confirm.seed_ledger_summary(prereg)
    assert first == confirm.seed_ledger_summary(prereg)
    assert first["streams"] == first["unique_seeds"] == 76456
    assert first["overlap_with_prior_streams"] == 0
