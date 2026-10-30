"""Encrypt and verify access-controlled full-development checkpoint seed bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MAGIC = b"QMBSEED1"
AAD = b"Quantum_Simulator QMutBench checkpoint seed v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(path: Path) -> bytes:
    value = path.read_text(encoding="ascii").strip()
    if len(value) != 64:
        raise ValueError("Seed key file must contain exactly 64 hexadecimal characters")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Seed key file is not hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError("Seed key must decode to 32 bytes")
    return raw


def _members(root: Path) -> list[Path]:
    required = [root / "full_campaign_manifest.json", root / "fold_checkpoints" /
                "fold_01_campaign_manifest.json"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Seed must contain campaign and fold-01 manifests")
    paths = sorted(path for path in (root / "fold_checkpoints").glob("fold_01_qnn_chunk_*.json")
                   if path.is_file())
    numbers = [int(path.stem.rsplit("_", 1)[1]) for path in paths]
    if numbers != list(range(1, 372)):
        raise ValueError("Seed must contain the contiguous, complete checkpoint history 1..371")
    if _sha256(paths[-1]) != "928cbbef8dc42afe43059dc8bff016aaa3b2f37370dc3440751d040444589298":
        raise ValueError("Checkpoint 371 SHA-256 differs from the preserved recovery digest")
    fold_manifest = json.loads(required[1].read_text(encoding="utf-8"))
    receipts = fold_manifest.get("chunks", [])
    if (len(receipts) < 371 or any(int(item.get("chunk", -1)) != i
                                   for i, item in enumerate(receipts, 1)) or
            any(item.get("status") != "completed" for item in receipts[:371]) or
            any(item.get("status") != "pending" for item in receipts[371:])):
        raise ValueError("Fold manifest must have 371 completed receipts followed only by pending chunks")
    for path, receipt in zip(paths, receipts[:371]):
        if receipt.get("checkpoint") != path.name or receipt.get("sha256") != _sha256(path):
            raise ValueError(f"Chunk receipt does not match checkpoint file: {path.name}")
    return [required[0], required[1], *paths]


def seal(source: Path, destination: Path, key_path: Path) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    source, destination = source.resolve(), destination.resolve()
    manifest = {path.relative_to(source).as_posix(): _sha256(path) for path in _members(source)}
    progress = json.loads((source / "full_campaign_manifest.json").read_text(encoding="utf-8"))
    if (progress.get("status") != "paused_checkpointed" or progress.get("frozen_test_accessed") is not False
            or progress.get("active_fold") != 1 or progress.get("active_chunk") != 371
            or progress.get("total_chunks_committed") != 371):
        raise ValueError("Refusing to seal: seed is not the verified fold-01 checkpoint 371 state")
    descriptor = {"schema": "qmutbench-checkpoint-seed-v1", "checkpoint": 371,
        "progress_manifest_sha256": manifest["full_campaign_manifest.json"],
        "member_sha256": manifest}
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite encrypted seed: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qmb-seed-") as temp:
        archive = Path(temp) / "seed.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            bundle.writestr("seed_manifest.json", json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
            for path in _members(source):
                bundle.write(path, path.relative_to(source).as_posix())
        nonce = os.urandom(12)
        encrypted = MAGIC + nonce + AESGCM(_key(key_path)).encrypt(nonce, archive.read_bytes(), AAD)
        with destination.open("xb") as output:
            output.write(encrypted)
            output.flush()
            os.fsync(output.fileno())
    return {"status": "sealed", "checkpoint": 371, "member_count": len(manifest),
            "ciphertext_sha256": _sha256(destination), "ciphertext_bytes": destination.stat().st_size,
            "seed_manifest_sha256": hashlib.sha256(json.dumps(descriptor, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()}


def open_bundle(encrypted_path: Path, destination: Path, key_path: Path,
                expected_sha256: str) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    encrypted_path, destination = encrypted_path.resolve(), destination.resolve()
    actual = _sha256(encrypted_path)
    if actual != expected_sha256:
        raise ValueError("Encrypted seed SHA-256 does not match its pinned digest")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to extract seed over an existing non-empty directory")
    raw = encrypted_path.read_bytes()
    if len(raw) < len(MAGIC) + 12 + 16 or raw[:len(MAGIC)] != MAGIC:
        raise ValueError("Encrypted seed header is invalid")
    nonce = raw[len(MAGIC):len(MAGIC) + 12]
    clear = AESGCM(_key(key_path)).decrypt(nonce, raw[len(MAGIC) + 12:], AAD)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qmb-open-") as temp:
        archive_path = Path(temp) / "seed.zip"
        archive_path.write_bytes(clear)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
                raise ValueError("Seed archive contains an unsafe path")
            descriptor = json.loads(archive.read("seed_manifest.json"))
            if descriptor.get("schema") != "qmutbench-checkpoint-seed-v1" or descriptor.get("checkpoint") != 371:
                raise ValueError("Seed archive manifest is incompatible")
            for name, digest in descriptor["member_sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"Seed member checksum mismatch: {name}")
            archive.extractall(destination)
    return {"status": "decrypted_and_verified", "checkpoint": 371,
            "ciphertext_sha256": actual, "member_count": len(descriptor["member_sha256"])}


def seal_directory(source: Path, destination: Path, key_path: Path) -> dict:
    """Encrypt a run artifact tree before it leaves a public GitHub runner."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    source, destination = source.resolve(), destination.resolve()
    paths = sorted(path for path in source.rglob("*") if path.is_file())
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("Run artifact directory is empty or contains a symbolic link")
    members = {path.relative_to(source).as_posix(): _sha256(path) for path in paths}
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite encrypted run artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {"schema": "qmutbench-run-artifact-v1", "member_sha256": members}
    with tempfile.TemporaryDirectory(prefix="qmb-run-seal-") as temp:
        archive_path = Path(temp) / "run.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("run_manifest.json", json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
            for path in paths:
                archive.write(path, path.relative_to(source).as_posix())
        nonce = os.urandom(12)
        ciphertext = MAGIC + nonce + AESGCM(_key(key_path)).encrypt(nonce, archive_path.read_bytes(), AAD)
        with destination.open("xb") as stream:
            stream.write(ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
    return {"status": "run_artifact_encrypted", "member_count": len(members),
            "ciphertext_sha256": _sha256(destination), "ciphertext_bytes": destination.stat().st_size}


def open_directory(encrypted_path: Path, destination: Path, key_path: Path) -> dict:
    """Authenticate and checksum every member before restoring a run artifact tree."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    encrypted_path, destination = encrypted_path.resolve(), destination.resolve()
    raw = encrypted_path.read_bytes()
    if len(raw) < len(MAGIC) + 12 + 16 or raw[:len(MAGIC)] != MAGIC:
        raise ValueError("Encrypted run artifact header is invalid")
    nonce = raw[len(MAGIC):len(MAGIC) + 12]
    clear = AESGCM(_key(key_path)).decrypt(nonce, raw[len(MAGIC) + 12:], AAD)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to extract run artifact over a non-empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qmb-run-open-") as temp:
        archive_path = Path(temp) / "run.zip"
        archive_path.write_bytes(clear)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                   for name in names):
                raise ValueError("Run artifact contains an unsafe path")
            descriptor = json.loads(archive.read("run_manifest.json"))
            if descriptor.get("schema") != "qmutbench-run-artifact-v1":
                raise ValueError("Run artifact manifest schema is incompatible")
            expected = descriptor.get("member_sha256", {})
            if set(names) != set(expected) | {"run_manifest.json"}:
                raise ValueError("Run artifact member list differs from its checksummed manifest")
            for name, digest in expected.items():
                info = archive.getinfo(name)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Run artifact contains a symbolic link")
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"Run artifact member checksum mismatch: {name}")
            for name in expected:
                target = destination.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(archive.read(name))
    return {"status": "run_artifact_decrypted_and_verified",
            "ciphertext_sha256": _sha256(encrypted_path), "member_count": len(expected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--source", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.add_argument("--key-file", type=Path, required=True)
    open_parser = sub.add_parser("open")
    open_parser.add_argument("--encrypted", type=Path, required=True)
    open_parser.add_argument("--output", type=Path, required=True)
    open_parser.add_argument("--key-file", type=Path, required=True)
    open_parser.add_argument("--sha256", required=True)
    seal_tree = sub.add_parser("seal-directory")
    seal_tree.add_argument("--source", type=Path, required=True)
    seal_tree.add_argument("--output", type=Path, required=True)
    seal_tree.add_argument("--key-file", type=Path, required=True)
    open_tree = sub.add_parser("open-directory")
    open_tree.add_argument("--encrypted", type=Path, required=True)
    open_tree.add_argument("--output", type=Path, required=True)
    open_tree.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(args.source, args.output, args.key_file)
    elif args.command == "open":
        result = open_bundle(args.encrypted, args.output, args.key_file, args.sha256)
    elif args.command == "seal-directory":
        result = seal_directory(args.source, args.output, args.key_file)
    else:
        result = open_directory(args.encrypted, args.output, args.key_file)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
