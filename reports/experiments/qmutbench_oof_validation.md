# QMutBench grouped OOF validation and error analysis

## Integrity and scope

Validated 578,117 OOF rows across 305 development origins against source SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`. OOF SHA-256: `7d203ac4cd6878a1caaf19eb768bb9a1e4f322b42fc56475cdc603591cc250ce`. Row identifiers, target values, origin groups, saved fold memberships, and finite probability bounds all match. Each fold's validation groups are disjoint from its training groups. The 70-origin historical test remains exploratory and was not evaluated.

## Pooled development OOF metrics

At saved threshold 0.485: macro F1 0.7566, balanced accuracy 0.7573, accuracy 0.7566, ROC-AUC 0.8491, average precision 0.8666, Brier score 0.1607.

Confusion matrix uses rows=true class [0,1] and columns=predicted class [0,1]: `[[215624, 62789], [77904, 221800]]`. Per-class precision, recall, F1, support and calibration bins are in the JSON companion.

OOF threshold 0.485 is read from the saved development model metadata. The nested grouped threshold audit selected thresholds using the other two folds and scored on the held fold; its average macro F1 is about 0.7557 versus 0.7548 at 0.5. Full-OOF threshold metrics are descriptive selection estimates, not unbiased evaluation.

## Slice findings

Tables below list the lowest macro-F1 slices with at least 100 rows. Small-support categories are omitted. Slice metrics use threshold 0.485.

### operation

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Replace | 120,676 | 0.731 | 0.723 | 0.776 | 0.871 | 19,061 | 9,792 |
| Add | 437,983 | 0.750 | 0.750 | 0.777 | 0.688 | 42,075 | 66,514 |
| Remove | 19,458 | 0.831 | 0.830 | 0.849 | 0.853 | 1,653 | 1,598 |

### gate

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| rxx | 35,248 | 0.589 | 0.588 | 0.745 | 0.919 | 7,785 | 1,999 |
| sx | 29,476 | 0.637 | 0.639 | 0.696 | 0.648 | 4,727 | 5,869 |
| z | 29,476 | 0.642 | 0.646 | 0.683 | 0.518 | 3,422 | 6,880 |
| p | 27,166 | 0.645 | 0.647 | 0.737 | 0.385 | 1,486 | 6,658 |
| h | 31,525 | 0.648 | 0.627 | 0.816 | 0.962 | 5,224 | 919 |
| rzz | 35,784 | 0.665 | 0.666 | 0.702 | 0.521 | 3,635 | 7,888 |
| rx | 29,744 | 0.666 | 0.660 | 0.731 | 0.863 | 5,961 | 2,572 |
| rz | 26,356 | 0.671 | 0.672 | 0.798 | 0.421 | 1,168 | 6,364 |
| cswap | 23,322 | 0.674 | 0.669 | 0.715 | 0.450 | 1,608 | 4,929 |
| swap | 23,539 | 0.684 | 0.685 | 0.714 | 0.597 | 2,741 | 4,617 |
| ry | 32,328 | 0.684 | 0.668 | 0.798 | 0.914 | 5,347 | 1,973 |
| x | 30,126 | 0.686 | 0.652 | 0.843 | 0.985 | 4,357 | 349 |

### algorithm

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| groundstatemedium | 3,396 | 0.481 | 0.692 | 0.973 | 0.478 | 39 | 1,556 |
| groundstatelarge | 4,634 | 0.490 | 0.700 | 0.974 | 0.498 | 54 | 2,049 |
| grover-noancilla | 10,095 | 0.538 | 0.546 | 0.677 | 0.563 | 1,727 | 2,808 |
| groundstatesmall | 364 | 0.550 | 0.687 | 0.942 | 0.502 | 9 | 146 |
| grover-v-chain | 2,537 | 0.577 | 0.577 | 0.660 | 0.645 | 503 | 537 |
| ae | 14,289 | 0.601 | 0.602 | 0.619 | 0.551 | 2,447 | 3,245 |
| portfolioqaoa | 7,706 | 0.603 | 0.600 | 0.757 | 0.954 | 1,679 | 251 |
| qaoa | 18,252 | 0.605 | 0.612 | 0.717 | 0.630 | 2,815 | 4,195 |
| pricingcall | 19,901 | 0.628 | 0.628 | 0.701 | 0.696 | 3,526 | 3,611 |
| tsp | 3,985 | 0.633 | 0.638 | 0.697 | 0.588 | 559 | 902 |
| qwalk-noancilla | 5,417 | 0.634 | 0.650 | 0.839 | 0.757 | 597 | 997 |
| routing | 4,058 | 0.642 | 0.654 | 0.510 | 0.642 | 881 | 511 |

### dominant_state

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| yes | 415,624 | 0.702 | 0.702 | 0.706 | 0.638 | 51,787 | 70,551 |
| no | 162,493 | 0.875 | 0.869 | 0.899 | 0.930 | 11,002 | 7,353 |

### qubit_bin

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| (-0.001, 4.0] | 22,295 | 0.673 | 0.673 | 0.719 | 0.714 | 3,544 | 3,620 |
| (4.0, 8.0] | 79,497 | 0.696 | 0.697 | 0.736 | 0.699 | 10,942 | 13,106 |
| (8.0, 16.0] | 324,218 | 0.735 | 0.735 | 0.748 | 0.696 | 37,303 | 48,208 |
| (16.0, 24.0] | 113,523 | 0.824 | 0.825 | 0.846 | 0.815 | 8,856 | 11,083 |
| (24.0, inf] | 38,584 | 0.885 | 0.884 | 0.915 | 0.924 | 2,144 | 1,887 |

### gate_count_bin

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| (100.0, 250.0] | 159,765 | 0.725 | 0.726 | 0.751 | 0.692 | 18,759 | 25,145 |
| (250.0, 500.0] | 202,195 | 0.740 | 0.739 | 0.738 | 0.671 | 21,618 | 29,823 |
| (50.0, 100.0] | 124,622 | 0.763 | 0.765 | 0.813 | 0.798 | 13,585 | 14,966 |
| (25.0, 50.0] | 60,982 | 0.806 | 0.805 | 0.836 | 0.850 | 6,011 | 5,383 |
| (-0.001, 25.0] | 30,553 | 0.819 | 0.819 | 0.840 | 0.852 | 2,816 | 2,587 |

### position_bin

| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| (100.0, 250.0] | 116,106 | 0.703 | 0.703 | 0.716 | 0.641 | 14,155 | 20,051 |
| (250.0, inf] | 50,862 | 0.741 | 0.740 | 0.758 | 0.655 | 4,853 | 8,020 |
| (50.0, 100.0] | 107,504 | 0.748 | 0.749 | 0.775 | 0.721 | 11,611 | 15,497 |
| (25.0, 50.0] | 119,416 | 0.766 | 0.767 | 0.805 | 0.767 | 12,325 | 15,469 |
| (10.0, 25.0] | 101,794 | 0.770 | 0.770 | 0.798 | 0.787 | 11,229 | 11,988 |
| (5.0, 10.0] | 39,431 | 0.801 | 0.800 | 0.811 | 0.820 | 4,040 | 3,782 |
| (-1.001, 5.0] | 43,004 | 0.821 | 0.821 | 0.802 | 0.857 | 4,576 | 3,097 |

## Origin variation and calibration

Across origins, mean macro F1 is 0.743 (median 0.717); mean accuracy is 0.776 (median 0.796). See the companion JSON for lowest-performing and largest-error origins, and equal-frequency reliability bins. Probabilities are not established as calibrated confidence.

The saved full-development estimator was loaded and repeated inference on an identical deterministic eight-row feature sample produced identical outputs. A direct equality comparison against each row's stored OOF prediction is not a valid reproducibility check: those predictions came from fold models that excluded the row's entire origin, while the saved estimator was fit on all development origins. No stored in-sample output from the final fitted estimator exists.

## Interpretation and next experiments

Error concentration in later circuit positions and larger circuits supports investigating position- and size-conditioned representations using the same grouped folds. Gate and algorithm slices should be checked for origin composition before feature changes, because the same origins generate many related records. Evaluate calibration only inside grouped development folds. These historical mutation outcome results do not establish hardware-noise detection or hardware-aware fault localization.
