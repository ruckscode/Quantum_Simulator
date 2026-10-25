# QMutBench dataset inspection

- Source: a local QMutBench CSV copy (read only; processed in 100,000-row chunks). The original download URL and license were not recorded with this inspection.
- File size: 59,913,911 bytes (about 57.1 MiB)
- Rows: 723,079 data rows (header excluded)
- Columns: 12

## Schema and missing values

| Column | Data type | Missing | Unique values (string columns) |
|---|---:|---:|---:|
| `Unnamed: 0` | `int64` | 0 | ? |
| `Origin_program` | `str` | 0 | 375 |
| `algorithm` | `str` | 0 | 28 |
| `qubits` | `int64` | 0 | ? |
| `gates` | `int64` | 0 | ? |
| `operation` | `str` | 0 | 3 |
| `Position_int` | `int64` | 0 | ? |
| `Killed` | `str` | 0 | 3 |
| `gate` | `str` | 0 | 19 |
| `algorithm_group` | `str` | 0 | 12 |
| `dominant_state` | `str` | 0 | 2 |
| `position` | `str` | 0 | 10 |

## Categorical/string value counts

### `Origin_program`

| Value | Rows |
|---|---:|
| `portfoliovqe_indep_qiskit_17` | 12,223 |
| `realamprandom_indep_qiskit_17` | 12,223 |
| `twolocalrandom_indep_qiskit_17` | 12,223 |
| `portfoliovqe_indep_qiskit_16` | 10,928 |
| `realamprandom_indep_qiskit_16` | 10,928 |
| `twolocalrandom_indep_qiskit_16` | 10,928 |
| `pricingput_indep_qiskit_15` | 10,089 |
| `portfoliovqe_indep_qiskit_15` | 9,705 |
| `realamprandom_indep_qiskit_15` | 9,705 |
| `twolocalrandom_indep_qiskit_15` | 9,705 |
| `pricingcall_indep_qiskit_15` | 9,531 |
| `su2random_indep_qiskit_16` | 8,944 |
| `portfoliovqe_indep_qiskit_14` | 8,554 |
| `realamprandom_indep_qiskit_14` | 8,554 |
| `twolocalrandom_indep_qiskit_14` | 8,554 |
| `su2random_indep_qiskit_15` | 7,845 |
| `portfoliovqe_indep_qiskit_13` | 7,475 |
| `realamprandom_indep_qiskit_13` | 7,475 |
| `twolocalrandom_indep_qiskit_13` | 7,475 |
| `su2random_indep_qiskit_14` | 6,818 |
| `portfoliovqe_indep_qiskit_12` | 6,468 |
| `realamprandom_indep_qiskit_12` | 6,468 |
| `twolocalrandom_indep_qiskit_12` | 6,468 |
| `pricingput_indep_qiskit_13` | 6,061 |
| `su2random_indep_qiskit_13` | 5,863 |
| `grover-noancilla_indep_qiskit_6` | 5,776 |
| `pricingcall_indep_qiskit_13` | 5,751 |
| `portfoliovqe_indep_qiskit_11` | 5,533 |
| `realamprandom_indep_qiskit_11` | 5,533 |
| `twolocalrandom_indep_qiskit_11` | 5,533 |
| `tsp_indep_qiskit_16` | 5,080 |
| `su2random_indep_qiskit_12` | 4,980 |
| `qwalk-noancilla_indep_qiskit_4` | 4,749 |
| `portfoliovqe_indep_qiskit_10` | 4,670 |
| `realamprandom_indep_qiskit_10` | 4,670 |
| `twolocalrandom_indep_qiskit_10` | 4,670 |
| `groundstate_large_indep_qiskit_14` | 4,634 |
| `portfolioqaoa_indep_qiskit_11` | 4,180 |
| `su2random_indep_qiskit_11` | 4,169 |
| `portfoliovqe_indep_qiskit_9` | 3,879 |
| `realamprandom_indep_qiskit_9` | 3,879 |
| `twolocalrandom_indep_qiskit_9` | 3,879 |
| `pricingput_indep_qiskit_11` | 3,759 |
| `pricingcall_indep_qiskit_11` | 3,573 |
| `grover-v-chain_indep_qiskit_7` | 3,531 |
| `portfolioqaoa_indep_qiskit_10` | 3,470 |
| `su2random_indep_qiskit_10` | 3,430 |
| `wstate_indep_qiskit_27` | 3,404 |
| `groundstate_medium_indep_qiskit_12` | 3,396 |
| `grover-noancilla_indep_qiskit_5` | 3,289 |
| `wstate_indep_qiskit_26` | 3,275 |
| `portfoliovqe_indep_qiskit_8` | 3,160 |
| `realamprandom_indep_qiskit_8` | 3,160 |
| `twolocalrandom_indep_qiskit_8` | 3,160 |
| `wstate_indep_qiskit_25` | 3,146 |
| `wstate_indep_qiskit_24` | 3,017 |
| `vqe_indep_qiskit_19` | 2,992 |
| `wstate_indep_qiskit_23` | 2,888 |
| `vqe_indep_qiskit_18` | 2,832 |
| `portfolioqaoa_indep_qiskit_9` | 2,826 |
| `tsp_indep_qiskit_9` | 2,805 |
| `su2random_indep_qiskit_9` | 2,763 |
| `wstate_indep_qiskit_22` | 2,759 |
| `vqe_indep_qiskit_17` | 2,672 |
| `wstate_indep_qiskit_21` | 2,630 |
| `pricingput_indep_qiskit_9` | 2,529 |
| `qgan_indep_qiskit_13` | 2,522 |
| `portfoliovqe_indep_qiskit_7` | 2,513 |
| `realamprandom_indep_qiskit_7` | 2,513 |
| `twolocalrandom_indep_qiskit_7` | 2,513 |
| `vqe_indep_qiskit_16` | 2,512 |
| `routing_indep_qiskit_12` | 2,508 |
| `wstate_indep_qiskit_20` | 2,501 |
| `qwalk-v-chain_indep_qiskit_13` | 2,418 |
| `wstate_indep_qiskit_19` | 2,372 |
| `vqe_indep_qiskit_15` | 2,352 |
| `qaoa_indep_qiskit_15` | 2,340 |
| `dj_indep_qiskit_28` | 2,296 |
| `qpeexact_indep_qiskit_25` | 2,282 |
| `qpeinexact_indep_qiskit_25` | 2,282 |
| `portfolioqaoa_indep_qiskit_8` | 2,248 |
| `wstate_indep_qiskit_18` | 2,243 |
| `pricingcall_indep_qiskit_9` | 2,219 |
| `dj_indep_qiskit_26` | 2,210 |
| `vqe_indep_qiskit_14` | 2,192 |
| `qaoa_indep_qiskit_14` | 2,184 |
| `qgan_indep_qiskit_12` | 2,184 |
| `qpeexact_indep_qiskit_24` | 2,177 |
| `qpeinexact_indep_qiskit_24` | 2,177 |
| `su2random_indep_qiskit_8` | 2,168 |
| `graphstate_indep_qiskit_29` | 2,146 |
| `qftentangled_indep_qiskit_25` | 2,145 |
| `wstate_indep_qiskit_17` | 2,114 |
| `qpeexact_indep_qiskit_23` | 2,096 |
| `qpeinexact_indep_qiskit_23` | 2,096 |
| `graphstate_indep_qiskit_28` | 2,072 |
| `qftentangled_indep_qiskit_24` | 2,071 |
| `qwalk-v-chain_indep_qiskit_11` | 2,068 |
| `vqe_indep_qiskit_13` | 2,032 |
| `qaoa_indep_qiskit_13` | 2,028 |
| `graphstate_indep_qiskit_27` | 1,998 |
| `qpeexact_indep_qiskit_22` | 1,991 |
| `qpeinexact_indep_qiskit_22` | 1,991 |
| `wstate_indep_qiskit_16` | 1,985 |
| `qftentangled_indep_qiskit_23` | 1,973 |
| `dj_indep_qiskit_24` | 1,938 |
| `portfoliovqe_indep_qiskit_6` | 1,938 |
| `realamprandom_indep_qiskit_6` | 1,938 |
| `twolocalrandom_indep_qiskit_6` | 1,938 |
| `graphstate_indep_qiskit_26` | 1,924 |
| `qpeexact_indep_qiskit_21` | 1,910 |
| `qpeinexact_indep_qiskit_21` | 1,910 |
| `qftentangled_indep_qiskit_22` | 1,899 |
| `ae_indep_qiskit_20` | 1,881 |
| `qaoa_indep_qiskit_12` | 1,872 |
| `vqe_indep_qiskit_12` | 1,872 |
| `qgan_indep_qiskit_11` | 1,870 |
| `wstate_indep_qiskit_15` | 1,856 |
| `graphstate_indep_qiskit_25` | 1,850 |
| `qpeexact_indep_qiskit_20` | 1,805 |
| `qpeinexact_indep_qiskit_20` | 1,805 |
| `qftentangled_indep_qiskit_21` | 1,801 |
| `dj_indep_qiskit_25` | 1,795 |
| `ae_indep_qiskit_19` | 1,783 |
| `graphstate_indep_qiskit_24` | 1,776 |
| `dj_indep_qiskit_27` | 1,757 |
| `portfolioqaoa_indep_qiskit_7` | 1,736 |
| `qftentangled_indep_qiskit_20` | 1,727 |
| `wstate_indep_qiskit_14` | 1,727 |
| `qpeexact_indep_qiskit_19` | 1,724 |
| `qpeinexact_indep_qiskit_19` | 1,724 |
| `qwalk-v-chain_indep_qiskit_9` | 1,718 |
| `qaoa_indep_qiskit_11` | 1,716 |
| `vqe_indep_qiskit_11` | 1,712 |
| `graphstate_indep_qiskit_23` | 1,702 |
| `ae_indep_qiskit_18` | 1,685 |
| `dj_indep_qiskit_22` | 1,666 |
| `dj_indep_qiskit_23` | 1,647 |
| `su2random_indep_qiskit_7` | 1,645 |
| `qftentangled_indep_qiskit_19` | 1,629 |
| `graphstate_indep_qiskit_22` | 1,628 |
| `qpeexact_indep_qiskit_18` | 1,619 |
| `qpeinexact_indep_qiskit_18` | 1,619 |
| `wstate_indep_qiskit_13` | 1,598 |
| `ae_indep_qiskit_17` | 1,587 |
| `pricingput_indep_qiskit_7` | 1,581 |
| `qgan_indep_qiskit_10` | 1,580 |
| `dj_indep_qiskit_21` | 1,561 |
| `qaoa_indep_qiskit_10` | 1,560 |
| `qftentangled_indep_qiskit_18` | 1,555 |
| `graphstate_indep_qiskit_21` | 1,554 |
| `vqe_indep_qiskit_10` | 1,552 |
| `qft_indep_qiskit_25` | 1,538 |
| `qpeexact_indep_qiskit_17` | 1,538 |
| `qpeinexact_indep_qiskit_17` | 1,538 |
| `pricingcall_indep_qiskit_7` | 1,519 |
| `grover-v-chain_indep_qiskit_5` | 1,507 |
| `ae_indep_qiskit_16` | 1,489 |
| `qft_indep_qiskit_24` | 1,488 |
| `graphstate_indep_qiskit_20` | 1,480 |
| `wstate_indep_qiskit_12` | 1,469 |
| `qftentangled_indep_qiskit_17` | 1,457 |
| `portfoliovqe_indep_qiskit_5` | 1,435 |
| `realamprandom_indep_qiskit_5` | 1,435 |
| `twolocalrandom_indep_qiskit_5` | 1,435 |
| `qpeexact_indep_qiskit_16` | 1,433 |
| `qpeinexact_indep_qiskit_16` | 1,433 |
| `qft_indep_qiskit_23` | 1,414 |
| `graphstate_indep_qiskit_19` | 1,406 |
| `qaoa_indep_qiskit_9` | 1,404 |
| `vqe_indep_qiskit_9` | 1,392 |
| `ae_indep_qiskit_15` | 1,391 |
| `qftentangled_indep_qiskit_16` | 1,383 |
| `qwalk-v-chain_indep_qiskit_7` | 1,368 |
| `qft_indep_qiskit_22` | 1,364 |
| `qpeexact_indep_qiskit_15` | 1,352 |
| `qpeinexact_indep_qiskit_15` | 1,352 |
| `wstate_indep_qiskit_11` | 1,340 |
| `dj_indep_qiskit_20` | 1,332 |
| `graphstate_indep_qiskit_18` | 1,332 |
| `qgan_indep_qiskit_9` | 1,314 |
| `ghz_indep_qiskit_30` | 1,297 |
| `ae_indep_qiskit_14` | 1,293 |
| `portfolioqaoa_indep_qiskit_6` | 1,290 |
| `qft_indep_qiskit_21` | 1,290 |
| `qftentangled_indep_qiskit_15` | 1,285 |
| `graphstate_indep_qiskit_17` | 1,258 |
| `ghz_indep_qiskit_29` | 1,254 |
| `qaoa_indep_qiskit_8` | 1,248 |
| `qpeexact_indep_qiskit_14` | 1,247 |
| `qpeinexact_indep_qiskit_14` | 1,247 |
| `qft_indep_qiskit_20` | 1,240 |
| `vqe_indep_qiskit_8` | 1,232 |
| `routing_indep_qiskit_6` | 1,218 |
| `ghz_indep_qiskit_28` | 1,211 |
| `qftentangled_indep_qiskit_14` | 1,211 |
| `wstate_indep_qiskit_10` | 1,211 |
| `dj_indep_qiskit_17` | 1,203 |
| `ae_indep_qiskit_13` | 1,195 |
| `su2random_indep_qiskit_6` | 1,194 |
| `graphstate_indep_qiskit_16` | 1,184 |
| `tsp_indep_qiskit_4` | 1,180 |
| `ghz_indep_qiskit_27` | 1,168 |
| `qft_indep_qiskit_19` | 1,166 |
| `qpeexact_indep_qiskit_13` | 1,166 |
| `qpeinexact_indep_qiskit_13` | 1,166 |
| `dj_indep_qiskit_19` | 1,165 |
| `dj_indep_qiskit_16` | 1,160 |
| `ghz_indep_qiskit_26` | 1,125 |
| `dj_indep_qiskit_18` | 1,122 |
| `qft_indep_qiskit_18` | 1,116 |
| `qftentangled_indep_qiskit_13` | 1,113 |
| `graphstate_indep_qiskit_15` | 1,110 |
| `ae_indep_qiskit_12` | 1,097 |
| `qaoa_indep_qiskit_7` | 1,092 |
| `ghz_indep_qiskit_25` | 1,082 |
| `wstate_indep_qiskit_9` | 1,082 |
| `qgan_indep_qiskit_8` | 1,072 |
| `vqe_indep_qiskit_7` | 1,072 |
| `qpeexact_indep_qiskit_12` | 1,061 |
| `qpeinexact_indep_qiskit_12` | 1,061 |
| `dj_indep_qiskit_15` | 1,055 |
| `qft_indep_qiskit_17` | 1,042 |
| `ghz_indep_qiskit_24` | 1,039 |
| `qftentangled_indep_qiskit_12` | 1,039 |
| `graphstate_indep_qiskit_14` | 1,036 |
| `qwalk-v-chain_indep_qiskit_5` | 1,018 |
| `portfoliovqe_indep_qiskit_4` | 1,004 |
| `realamprandom_indep_qiskit_4` | 1,004 |
| `twolocalrandom_indep_qiskit_4` | 1,004 |
| `ae_indep_qiskit_11` | 999 |
| `ghz_indep_qiskit_23` | 996 |
| `qft_indep_qiskit_16` | 992 |
| `qpeexact_indep_qiskit_11` | 980 |
| `qpeinexact_indep_qiskit_11` | 980 |
| `dj_indep_qiskit_13` | 969 |
| `graphstate_indep_qiskit_13` | 962 |
| `ghz_indep_qiskit_22` | 953 |
| `wstate_indep_qiskit_8` | 953 |
| `dj_indep_qiskit_14` | 950 |
| `qftentangled_indep_qiskit_11` | 941 |
| `qaoa_indep_qiskit_6` | 936 |
| `dj_indep_qiskit_12` | 926 |
| `qft_indep_qiskit_15` | 918 |
| `vqe_indep_qiskit_6` | 912 |
| `ghz_indep_qiskit_21` | 910 |
| `portfolioqaoa_indep_qiskit_5` | 910 |
| `ae_indep_qiskit_10` | 901 |
| `graphstate_indep_qiskit_12` | 888 |
| `pricingcall_indep_qiskit_5` | 881 |
| `pricingput_indep_qiskit_5` | 881 |
| `qpeexact_indep_qiskit_10` | 875 |
| `qpeinexact_indep_qiskit_10` | 875 |
| `qft_indep_qiskit_14` | 868 |
| `ghz_indep_qiskit_20` | 867 |
| `qftentangled_indep_qiskit_10` | 867 |
| `qgan_indep_qiskit_7` | 854 |
| `ghz_indep_qiskit_19` | 824 |
| `wstate_indep_qiskit_7` | 824 |
| `su2random_indep_qiskit_5` | 815 |
| `graphstate_indep_qiskit_11` | 814 |
| `ae_indep_qiskit_9` | 803 |
| `qft_indep_qiskit_13` | 794 |
| `qpeexact_indep_qiskit_9` | 794 |
| `qpeinexact_indep_qiskit_9` | 794 |
| `ghz_indep_qiskit_18` | 781 |
| `qaoa_indep_qiskit_5` | 780 |
| `qftentangled_indep_qiskit_9` | 769 |
| `vqe_indep_qiskit_5` | 752 |
| `qft_indep_qiskit_12` | 744 |
| `graphstate_indep_qiskit_10` | 740 |
| `ghz_indep_qiskit_17` | 738 |
| `ae_indep_qiskit_8` | 705 |
| `dj_indep_qiskit_11` | 697 |
| `ghz_indep_qiskit_16` | 695 |
| `qftentangled_indep_qiskit_8` | 695 |
| `wstate_indep_qiskit_6` | 695 |
| `qpeexact_indep_qiskit_8` | 689 |
| `qpeinexact_indep_qiskit_8` | 689 |
| `qft_indep_qiskit_11` | 670 |
| `qwalk-noancilla_indep_qiskit_3` | 668 |
| `qwalk-v-chain_indep_qiskit_3` | 668 |
| `graphstate_indep_qiskit_9` | 666 |
| `qgan_indep_qiskit_6` | 660 |
| `dj_indep_qiskit_10` | 654 |
| `grover-noancilla_indep_qiskit_4` | 654 |
| `grover-v-chain_indep_qiskit_4` | 654 |
| `ghz_indep_qiskit_15` | 652 |
| `portfoliovqe_indep_qiskit_3` | 645 |
| `realamprandom_indep_qiskit_3` | 645 |
| `twolocalrandom_indep_qiskit_3` | 645 |
| `qaoa_indep_qiskit_4` | 624 |
| `qft_indep_qiskit_10` | 620 |
| `ghz_indep_qiskit_14` | 609 |
| `qpeexact_indep_qiskit_7` | 608 |
| `qpeinexact_indep_qiskit_7` | 608 |
| `qftentangled_indep_qiskit_7` | 597 |
| `portfolioqaoa_indep_qiskit_4` | 596 |
| `graphstate_indep_qiskit_8` | 592 |
| `vqe_indep_qiskit_4` | 592 |
| `ghz_indep_qiskit_13` | 566 |
| `wstate_indep_qiskit_5` | 566 |
| `qft_indep_qiskit_9` | 546 |
| `ghz_indep_qiskit_12` | 523 |
| `qftentangled_indep_qiskit_6` | 523 |
| `graphstate_indep_qiskit_7` | 518 |
| `ae_indep_qiskit_6` | 509 |
| `su2random_indep_qiskit_4` | 508 |
| `dj_indep_qiskit_8` | 506 |
| `qpeexact_indep_qiskit_6` | 503 |
| `qpeinexact_indep_qiskit_6` | 503 |
| `qft_indep_qiskit_8` | 496 |
| `qgan_indep_qiskit_5` | 490 |
| `dj_indep_qiskit_9` | 487 |
| `ghz_indep_qiskit_11` | 480 |
| `qaoa_indep_qiskit_3` | 468 |
| `dj_indep_qiskit_7` | 463 |
| `graphstate_indep_qiskit_6` | 444 |
| `ghz_indep_qiskit_10` | 437 |
| `wstate_indep_qiskit_4` | 437 |
| `vqe_indep_qiskit_3` | 432 |
| `qftentangled_indep_qiskit_5` | 425 |
| `qft_indep_qiskit_7` | 422 |
| `qpeexact_indep_qiskit_5` | 422 |
| `qpeinexact_indep_qiskit_5` | 422 |
| `ae_indep_qiskit_5` | 411 |
| `ghz_indep_qiskit_9` | 394 |
| `qft_indep_qiskit_6` | 372 |
| `graphstate_indep_qiskit_5` | 370 |
| `groundstate_small_indep_qiskit_4` | 364 |
| `dj_indep_qiskit_6` | 358 |
| `ghz_indep_qiskit_8` | 351 |
| `qftentangled_indep_qiskit_4` | 351 |
| `portfolioqaoa_indep_qiskit_3` | 348 |
| `qgan_indep_qiskit_4` | 344 |
| `dj_indep_qiskit_4` | 334 |
| `realamprandom_indep_qiskit_2` | 332 |
| `routing_indep_qiskit_2` | 332 |
| `twolocalrandom_indep_qiskit_2` | 332 |
| `qpeexact_indep_qiskit_4` | 317 |
| `qpeinexact_indep_qiskit_4` | 317 |
| `dj_indep_qiskit_5` | 315 |
| `ae_indep_qiskit_4` | 313 |
| `ghz_indep_qiskit_7` | 308 |
| `wstate_indep_qiskit_3` | 308 |
| `qft_indep_qiskit_5` | 298 |
| `graphstate_indep_qiskit_4` | 296 |
| `grover-noancilla_indep_qiskit_3` | 284 |
| `grover-v-chain_indep_qiskit_3` | 284 |
| `su2random_indep_qiskit_3` | 273 |
| `ghz_indep_qiskit_6` | 265 |
| `qftentangled_indep_qiskit_3` | 253 |
| `qft_indep_qiskit_4` | 248 |
| `qpeexact_indep_qiskit_3` | 236 |
| `qpeinexact_indep_qiskit_3` | 236 |
| `dj_indep_qiskit_3` | 229 |
| `ghz_indep_qiskit_5` | 222 |
| `graphstate_indep_qiskit_3` | 222 |
| `qgan_indep_qiskit_3` | 222 |
| `ae_indep_qiskit_3` | 215 |
| `ghz_indep_qiskit_4` | 179 |
| `qft_indep_qiskit_3` | 174 |
| `qftentangled_indep_qiskit_2` | 165 |
| `wstate_indep_qiskit_2` | 165 |
| `ghz_indep_qiskit_3` | 136 |
| `qpeexact_indep_qiskit_2` | 121 |
| `qpeinexact_indep_qiskit_2` | 121 |
| `qft_indep_qiskit_2` | 114 |
| `qgan_indep_qiskit_2` | 114 |
| `ae_indep_qiskit_2` | 107 |
| `su2random_indep_qiskit_2` | 100 |
| `grover-noancilla_indep_qiskit_2` | 92 |
| `grover-v-chain_indep_qiskit_2` | 92 |
| `ghz_indep_qiskit_2` | 85 |
| `dj_indep_qiskit_2` | 56 |

### `algorithm`

| Value | Rows |
|---|---:|
| `realamprandom` | 80,462 |
| `twolocalrandom` | 80,462 |
| `portfoliovqe` | 80,130 |
| `su2random` | 51,515 |
| `wstate` | 46,565 |
| `graphstate` | 31,968 |
| `vqe` | 29,104 |
| `qpeexact` | 28,946 |
| `qpeinexact` | 28,946 |
| `dj` | 28,851 |
| `qftentangled` | 27,874 |
| `pricingput` | 24,900 |
| `pricingcall` | 23,474 |
| `ghz` | 20,147 |
| `qft` | 19,934 |
| `ae` | 18,364 |
| `qaoa` | 18,252 |
| `portfolioqaoa` | 17,604 |
| `qgan` | 13,226 |
| `grover-noancilla` | 10,095 |
| `qwalk-v-chain` | 9,258 |
| `tsp` | 9,065 |
| `grover-v-chain` | 6,068 |
| `qwalk-noancilla` | 5,417 |
| `groundstatelarge` | 4,634 |
| `routing` | 4,058 |
| `groundstatemedium` | 3,396 |
| `groundstatesmall` | 364 |

### `operation`

| Value | Rows |
|---|---:|
| `Add` | 548,733 |
| `Replace` | 149,972 |
| `Remove` | 24,374 |

### `Killed`

| Value | Rows |
|---|---:|
| `WOO` | 368,568 |
| `Survivor` | 348,299 |
| `OPO` | 6,212 |

### `gate`

| Value | Rows |
|---|---:|
| `cx` | 54,561 |
| `rzz` | 45,232 |
| `rxx` | 44,050 |
| `ccx` | 40,935 |
| `ry` | 40,444 |
| `h` | 39,399 |
| `x` | 37,582 |
| `cz` | 37,557 |
| `rx` | 37,111 |
| `id` | 36,814 |
| `s` | 36,814 |
| `sx` | 36,814 |
| `y` | 36,814 |
| `z` | 36,814 |
| `t` | 36,654 |
| `p` | 33,912 |
| `rz` | 32,887 |
| `swap` | 29,491 |
| `cswap` | 29,194 |

### `algorithm_group`

| Value | Rows |
|---|---:|
| `vqe` | 343,190 |
| `ae` | 66,738 |
| `qpe` | 57,892 |
| `qft` | 47,808 |
| `wstate` | 46,565 |
| `qaoa` | 35,856 |
| `graphstate` | 31,968 |
| `dj` | 28,851 |
| `ghz` | 20,147 |
| `grover` | 16,163 |
| `qwalk` | 14,675 |
| `qgan` | 13,226 |

### `dominant_state`

| Value | Rows |
|---|---:|
| `yes` | 519,839 |
| `no` | 203,240 |

### `position`

| Value | Rows |
|---|---:|
| `91-100%` | 95,343 |
| `11-20%` | 75,341 |
| `0-10%` | 74,958 |
| `21-30%` | 73,901 |
| `81-90%` | 70,309 |
| `31-40%` | 69,502 |
| `41-50%` | 67,166 |
| `71-80%` | 66,790 |
| `61-70%` | 65,345 |
| `51-60%` | 64,424 |

## Mutation and circuit findings

- Mutation operation counts (`operation`): `Add` 548,733; `Remove` 24,374; `Replace` 149,972
- Unique original circuits (`Origin_program`): 375.
- Mutant records: 723,079 rows; the file has no separate mutant identifier column.
- Outcome/survival fields: `Killed` is present with counts `OPO` 6,212; `Survivor` 348,299; `WOO` 368,568. This is an outcome-like categorical field; labels are preserved as encoded, not interpreted.
- QASM retrieval clues: `Origin_program` holds circuit-name identifiers (for example, `portfoliovqe_indep_qiskit_17`), not explicit filesystem paths. No mutant-specific path or filename column is present. `algorithm`, `qubits`, `operation`, `gate`, and position fields describe records and may help identify/filter the corresponding QASM group, but the CSV alone does not provide a direct mutant file path.
