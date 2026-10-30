# QMutBench QASM feature and model audit

## Data and integrity

- Verified official ingestion metadata SHA-256: `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`; row-value comparison against `data/qmutbench_full/data/all_data.csv` is `VALUE_IDENTICAL` over 723,079 rows (the files differ in byte size/hash because of line endings).
- Grouped development OOF: 578,117 unique record IDs across 305 origins; 3 folds with record counts {'3': 192111, '1': 194269, '2': 191737}; cross-fold origin count 0.
- QASM structure cache: 578,117 rows, 58 chunks, zero parse errors, all chunk checksums verified.
- Mutation-difference cache: 578,117 rows, 58 chunks, zero parse errors, all chunk checksums verified.
- Existing binary mapping verified record-by-record: WOO=1, OPO=1, Survivor=0; disagreements: 0.
- Frozen-test outcomes were not opened. Its independent file checksum was not rehashed in this turn because no frozen-test artifact/hash was located in the available manifest.

## Saved feature comparison results

All QASM comparison rows below are historical 3-fold grouped OOF screens on the same 578,117 records. They are not directly comparable with the separate 5-fold production champion because the fold contracts differ.

| Candidate | Feature family | Accuracy | Macro F1 | Balanced accuracy | Macro F1 change vs matched QASM baseline | Decision |
|---|---|---:|---:|---:|---:|---|
| 3-fold metadata baseline | Metadata | 76.63% | 76.63% | 76.75% | reference | — |
| 3-fold QASM mutant gate composition | Metadata + gate-family fractions | 77.11% | 77.11% | 77.17% | +0.48 pp | Two of three fold deltas positive; not consistent |
| 3-fold QASM structure only | Structural counts/depth | 75.95% | 75.95% | 76.07% | -0.68 pp | Lower than metadata baseline |
| Position-fraction + mutant gate fractions | Metadata + mutant gate-family composition | 77.15% | 77.14% | 77.19% | +0.32 pp | Two of three fold deltas positive; not promoted |
| Position-fraction + delta gate composition | Metadata + mutant-minus-original fractions | 76.40% | 76.40% | 76.54% | -0.23 pp | Regressed in all three folds |
| All registered QASM features | Combined | — | — | — | — | Partial: fold 1 only; no pooled score |

### Five-fold historical reference

The existing mutation-aware ExtraTrees champion has saved grouped 5-fold OOF metrics of **79.33% accuracy, 79.33% macro F1, and 79.41% balanced accuracy** on the same 578,117 development records / 305 origins. Its matched five-fold baseline scored 77.34% accuracy and 77.34% macro F1. QASM screens above use different three-fold assignments, so they do not establish improvement over this champion. The 85% accuracy research target remains 5.67 percentage points above the verified five-fold champion.

## Leakage and feature handling

`source_gate_count_match` is emitted in the feature CSV as a diagnostic but omitted from the manifest’s registered model feature list. It is not constant: counts by label are `{'label=1,match=1': 200891, 'label=0,match=1': 217275, 'label=1,match=0': 98813, 'label=0,match=0': 61138}`. This association warrants a same-fold ablation and provenance review; it does not alone prove target leakage. The field remains excluded. Identifiers, group names, paths, and folds were not used as predictive columns.

## QML status

No full-data QMutBench binary QNN or hybrid fit was run in this continuation. Existing QNN scores on the 205-row three-class subset are a separate target and are not compared with the binary champion. The local `.venv` has PennyLane and Qiskit but no CUDA GPU; memory remained below the 1 GiB guard.
