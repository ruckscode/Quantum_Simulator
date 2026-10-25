# End-to-End Diagnosis Report

## Test scenarios

- NORMAL circuit execution with injected NORMAL classifier; verified no localization.
- PROGRAM_FAULT execution with gate replacement, gate deletion, gate insertion, parameter modification, and two-qubit gate fault; verified the existing program localizer includes the injected operation/qubits.
- HARDWARE_ANOMALY execution with readout error, gate error, T1/T2 degradation, and combined hardware degradation; verified the existing hardware localizer includes the affected component.
- Hardware-induced trace deviation with identical expected and observed circuit; verified trace-only candidates remain separate from program-fault evidence.
- Verified mock classification and localization are separate layers and existing rule-based diagnosis/localizers are used.

## Actual results

- New end-to-end tests: **12 passed**.
- End-to-end plus Stage 1/2/3 diagnosis/localization selection: **93 passed**.
- Full maintained test suite: **358 passed**.
- All classifier predictions in these scenarios came from deterministic injected mocks. These counts are pipeline architecture results and do not measure ML accuracy.

## Classification pipeline status

- Observable feature extraction and classifier adapter interface are exercised end to end using the approved five observable features.
- Class routing works for NORMAL, PROGRAM_FAULT, and HARDWARE_ANOMALY with mocked predictions.
- The real model backend is unavailable; no model accuracy is reported.

## Localization pipeline status

- Existing program localization identified the injected operation and qubits for the five controlled program-fault types.
- Existing hardware localization identified the affected hardware component for the four controlled hardware scenarios.
- NORMAL produced no positive localization. Trace deviation alone did not become program-fault evidence.
- Existing rule-based diagnosis and localization regression tests passed.

## End-to-end integration status

- The execution → observable features → classifier interface → broad diagnosis class → existing localization → final diagnosis pipeline is complete and tested at the integration architecture level.

## Limitations caused by unavailable real ML model

- The genuine persisted sklearn model artifact is unavailable because Windows Application Control blocks sklearn's native `_predictor` DLL.
- Mock classifier outputs validate routing only; they do not establish real model predictions, accuracy, calibration, or generalization.
- The evaluation uses the project simulator and hardware model; it is not real-hardware validation.
