# Current Project Limitations

- Real-data diagnosis accuracy is not available because the audited Bugs4Q and QMutBench outputs do not include compatible independent diagnosis labels.
- Normal-circuit observations still need a reproducible simulator-generated baseline workflow.
- The 97% score came from generated synthetic data and must not be presented as project or benchmark accuracy.
- IBM hardware validation remains pending because authentication/instance access previously failed.
- Hardware fault localization requires real calibration/execution evidence for external validation.
- Full benchmark execution is constrained by supported operations, runtime, and dataset availability.

## Safe next steps

1. Generate baseline observations from the local simulator.
2. Keep controlled injected-fault labels explicitly separate from official benchmark labels.
3. Evaluate with circuit-family-separated train/test splits.
4. Report accuracy only for cases with valid labels and disclose the data source.
5. Run IBM validation only after account access works and the user explicitly authorizes job submission.
