# Mute-Alg full compatibility recheck — 2026-10-30

The restricted Qiskit 1.4.5 loader scanned all 169 upstream `qcs.pkl` files from the local Mute-Alg snapshot. All 169 loaded and normalized without unreadable files. Pair-order semantics were retained; no labels were changed.

- 16,360 circuit rows were read (8,180 original/buggy pairs).
- 3,610 complete pairs (7,220 rows) were excluded because exact OpenQASM fingerprints had conflicting labels.
- 9,140 rows remain, balanced 4,570/4,570, grouped into 812 duplicate-safe components and five equal folds.
- The frozen test was not accessed.
- Input SHA-256 `d7a7010bfc6421a94b5cb03b70568cfe7fd2537c2ac1f11ae433e29b4a42e853` and fold SHA-256 `d459f421e043ab4e7cc4194a22a11e759a7fbe9a4a4447e766324484bff106b3` exactly match the existing validated artifact and task contract.

The scan wrote a fresh reproduction under this campaign folder. The script stopped at its deliberate existing-contract guard rather than overwriting the canonical contract; the identical existing contract remains valid. This confirms the compatibility-selected subset is trainable and reproducible, but does not remove the contradictory-label limitation for the excluded source pairs.
