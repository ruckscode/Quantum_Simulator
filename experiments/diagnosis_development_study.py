"""Fail-closed preflight helpers for a future bounded diagnosis pilot.

This module validates a supplied independent family frame and expands its
frozen condition matrix. It does not generate circuits or run simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from experiments.diagnosis_lineage_inventory import SAMPLER_VERSION, verify_hash


class StudyBlockedError(ValueError):
    """Raised when lineage or partition evidence cannot support a clean study."""


class ResourceLimitExceeded(RuntimeError):
    """Raised before reserving work that would exceed the study budget."""


def audit_materialized_sampling_frame(
    inventory: Mapping[str, Any], frame: Mapping[str, Any], *,
    required_development: int = 8, required_validation: int = 8,
    verified_reconciliation_attestation: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Verify persisted inventory/frame integrity before any future generation.

    Hash integrity is necessary but not sufficient: inventory completeness must
    be independently established and explicitly represented in the document.
    """
    if not verify_hash(dict(inventory), "inventory_sha256"):
        raise StudyBlockedError("prior inventory hash is missing or invalid")
    if not verify_hash(dict(frame), "frame_sha256"):
        raise StudyBlockedError("sampling frame hash is missing or invalid")
    if frame.get("sampler_version") != SAMPLER_VERSION:
        raise StudyBlockedError("sampling frame uses an unsupported sampler version")
    if frame.get("inventory_sha256") != inventory.get("inventory_sha256"):
        raise StudyBlockedError("sampling frame does not match the recorded prior inventory")
    if inventory.get("complete") is not True:
        raise StudyBlockedError("prior circuit inventory is not certified complete; cannot exclude all prior descendants")
    summary = inventory.get("summary")
    attestation = verified_reconciliation_attestation
    if (
        not isinstance(summary, Mapping)
        or summary.get("unknown_lineage_circuit_count") != 0
        or summary.get("conflict_count") != 0
        or inventory.get("missing_scan_roots")
        or not isinstance(attestation, Mapping)
        or attestation.get("status") != "VERIFIED_COMPLETE"
        or attestation.get("inventory_sha256") != inventory.get("inventory_sha256")
        or attestation.get("frame_sha256") != frame.get("frame_sha256")
        or not _is_sha256(attestation.get("reconciliation_sha256"))
        or not _is_sha256(attestation.get("attestation_sha256"))
        or not isinstance(attestation.get("verified_by"), str)
        or not attestation.get("verified_by", "").strip()
    ):
        raise StudyBlockedError(
            "prior inventory lacks zero-unresolved metrics and a hash-bound, externally verified inventory/frame reconciliation attestation"
        )
    if frame.get("status") != "ready":
        raise StudyBlockedError(str(frame.get("blocker") or "sampling frame is not ready"))
    candidates = frame.get("candidates")
    if not isinstance(candidates, list):
        raise StudyBlockedError("sampling frame candidates must be a list")
    try:
        counts = audit_family_partitions(candidates, inventory)
    except (KeyError, TypeError, AttributeError) as exc:
        raise StudyBlockedError(f"sampling frame contains malformed lineage records: {exc}") from exc
    if counts["development"] < required_development or counts["validation"] < required_validation:
        raise StudyBlockedError(
            f"sampling frame has {counts['development']} development and {counts['validation']} validation roots; "
            f"requires {required_development} and {required_validation}"
        )
    selected = frame.get("selected")
    if not isinstance(selected, dict) or any(
        len(selected.get(partition, [])) != required
        for partition, required in (("development", required_development), ("validation", required_validation))
    ):
        raise StudyBlockedError("sampling frame selected roots do not match required partition sizes")
    return counts


def audit_family_partitions(
    family_frame: Sequence[Mapping[str, Any]],
    prior_inventory: Mapping[str, Any],
) -> dict[str, int]:
    """Require complete ancestry inventory and disjoint new family roots."""

    if prior_inventory.get("complete") is not True:
        raise StudyBlockedError(
            "prior circuit inventory is not certified complete; cannot exclude all prior descendants"
        )
    prior = prior_inventory.get("families")
    if not isinstance(prior, list) or not prior:
        raise StudyBlockedError("complete prior inventory must list known family roots")
    prior_roots: set[str] = set()
    prior_hashes: set[str] = set()
    for entry in prior:
        root = entry.get("lineage_root_id")
        fingerprint = entry.get("topology_sha256")
        if not isinstance(root, str) or not root or not _is_sha256(fingerprint):
            raise StudyBlockedError("prior inventory entries need lineage_root_id and topology_sha256")
        prior_roots.add(root)
        prior_hashes.add(fingerprint)

    seen_ids: set[str] = set()
    seen_roots: set[str] = set()
    seen_hashes: set[str] = set()
    partitions: dict[str, set[str]] = {"development": set(), "validation": set()}
    for family in family_frame:
        family_id = family.get("family_id")
        root = family.get("lineage_root_id")
        fingerprint = family.get("topology_sha256")
        partition = family.get("partition")
        ancestors = family.get("ancestor_root_ids")
        if not all(isinstance(item, str) and item for item in (family_id, root)):
            raise StudyBlockedError("each family needs nonempty family_id and lineage_root_id")
        if not _is_sha256(fingerprint):
            raise StudyBlockedError(f"family {family_id!r} lacks a canonical topology SHA-256")
        if partition not in partitions:
            raise StudyBlockedError(f"family {family_id!r} has invalid partition {partition!r}")
        if not isinstance(ancestors, list) or any(not isinstance(item, str) for item in ancestors):
            raise StudyBlockedError(f"family {family_id!r} needs explicit ancestor_root_ids")
        if family_id in seen_ids or root in seen_roots or fingerprint in seen_hashes:
            raise StudyBlockedError("family IDs, lineage roots, and topology fingerprints must be unique")
        if root in prior_roots or set(ancestors) & prior_roots:
            raise StudyBlockedError(f"family {family_id!r} descends from a previously inspected lineage root")
        if fingerprint in prior_hashes:
            raise StudyBlockedError(f"family {family_id!r} duplicates a previously inspected topology")
        seen_ids.add(family_id)
        seen_roots.add(root)
        seen_hashes.add(fingerprint)
        partitions[partition].add(root)

    if not partitions["development"] or not partitions["validation"]:
        raise StudyBlockedError("development and validation partitions must both contain families")
    if partitions["development"] & partitions["validation"]:
        raise StudyBlockedError("lineage roots cross development and validation partitions")
    for family in family_frame:
        if set(family["ancestor_root_ids"]) & seen_roots:
            raise StudyBlockedError(
                f"family {family['family_id']!r} is a descendant of another proposed family root"
            )
    return {name: len(roots) for name, roots in partitions.items()}


def build_condition_manifest(
    family_frame: Sequence[Mapping[str, Any]],
    *,
    strata: Sequence[str],
    shot_counts: Sequence[int],
    repetitions: int,
    master_seed: int,
) -> list[dict[str, Any]]:
    """Expand a validated family frame deterministically (no simulator runs)."""

    if not strata or len(set(strata)) != len(strata) or any(not item for item in strata):
        raise ValueError("strata must be nonempty and unique")
    if not shot_counts or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shot_counts):
        raise ValueError("shot_counts must contain positive integers")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError("master_seed must be an integer")

    rows: list[dict[str, Any]] = []
    for family in sorted(family_frame, key=lambda item: str(item["family_id"])):
        for stratum in strata:
            for shots in shot_counts:
                for repetition in range(repetitions):
                    seed_material = (
                        f"{master_seed}|{family['lineage_root_id']}|{stratum}|{shots}|{repetition}"
                    ).encode("utf-8")
                    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
                    rows.append({
                        "family_id": family["family_id"],
                        "lineage_root_id": family["lineage_root_id"],
                        "partition": family["partition"],
                        "stratum": stratum,
                        "shots": shots,
                        "repetition": repetition,
                        "seed": seed,
                    })
    return rows


@dataclass(frozen=True)
class StudyResourceLimits:
    max_circuit_executions: int
    max_shot_executions: int
    max_runtime_seconds: float
    max_peak_memory_bytes: int
    max_output_bytes: int


class ResourceBudget:
    """Atomic budget guard; an over-limit reservation is rejected unchanged."""

    def __init__(self, limits: StudyResourceLimits) -> None:
        self.limits = limits
        self.circuit_executions = 0
        self.shot_executions = 0
        self.output_bytes = 0

    def reserve(self, *, circuit_executions: int, shot_executions: int, output_bytes: int) -> None:
        values = (circuit_executions, shot_executions, output_bytes)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("resource reservations must be nonnegative integers")
        proposed = (
            self.circuit_executions + circuit_executions,
            self.shot_executions + shot_executions,
            self.output_bytes + output_bytes,
        )
        maxima = (
            self.limits.max_circuit_executions,
            self.limits.max_shot_executions,
            self.limits.max_output_bytes,
        )
        if any(value > maximum for value, maximum in zip(proposed, maxima)):
            raise ResourceLimitExceeded("study reservation exceeds its preregistered resource budget")
        self.circuit_executions, self.shot_executions, self.output_bytes = proposed

    def check_observed(self, *, runtime_seconds: float, peak_memory_bytes: int) -> None:
        if runtime_seconds > self.limits.max_runtime_seconds:
            raise ResourceLimitExceeded("study exceeded its preregistered runtime limit")
        if peak_memory_bytes > self.limits.max_peak_memory_bytes:
            raise ResourceLimitExceeded("study exceeded its preregistered memory limit")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
