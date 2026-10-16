# Bugs4Q official-source integration

The Bugs4Q source-backed integration is optional because the official source
dataset is external and is not bundled with every repository checkout.

## Expected location

From the repository root, the adapter expects the official Qiskit source cases
at:

```text
data/Bugs4Q_official/extracted/Zenodo-Framework/Bugs4Q-Framework/qiskit/
  1/buggy_1.py
  1/fixed_1.py
  1/modify_1.txt
  ...
  42/buggy_42.py
  42/fixed_42.py
  42/modify_42.txt
```

The dataset also contains additional test and metadata files. Preserve the
original directory and file names when extracting it. Obtain the archive from
the official dataset/project distribution and follow its terms and attribution
requirements; do not substitute generated fixtures for official cases.

## Test behavior

Run the regular test suite from the repository root:

```bash
python -m pytest -q --import-mode=importlib
```

When all 42 official cases are present, the source-backed integration tests run
their translation, execution, measurement-mapping, and reporting assertions.
When the official files are absent or incomplete, those tests are reported as
skipped with an explanatory reason. This allows a clean clone to validate the
local simulator and the dataset-independent Bugs4Q adapter tests without
misreporting missing external data as a simulator defect.

A skip is not a successful Bugs4Q benchmark run. To claim source-backed
integration coverage, install the complete dataset and rerun the suite. The
adapter's benchmark report should continue to identify missing data explicitly.
