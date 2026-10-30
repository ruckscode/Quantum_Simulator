# Test report — current continuation

- Requested `tests/test_qmutbench_full_corpus_streaming.py`: **not present** in this checkout; `experiments/qmutbench_full_corpus_streaming.py` is also absent. The repository’s existing `experiments/qmutbench_qasm_features.py` and completed cache were used instead; no duplicate extractor was introduced.
- Initial system-Python pytest collection failed because repository modules were not on its import path. The first `.venv` attempt also hit sandbox-protected `%TEMP%`. These were environment invocation issues, not assertion failures.
- Rerun with temp/basetemp inside the writable repository: `tests/test_qmutbench_qasm_features.py`, `tests/test_qmutbench_qasm_mutation_diff.py`, and `tests/test_qmutbench_corpus_pipeline.py`: **21 passed**.
- Broader focused suite: `tests/test_qmutbench_master_optimization.py`, `tests/test_qmutbench_learning_curve.py`, `tests/test_qmutbench_selected_pipeline.py`, and `tests/test_qmutbench_ml_pipeline_resume.py`: **19 passed**.
- Total relevant passing tests: **40**. No test used frozen-test labels.

Reproduction (PowerShell, repository root):

```powershell
$env:TEMP = Join-Path (Get-Location) '.tmp-qmutbench-tests'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest --basetemp="$PWD\.pytest-full-corpus-campaign" tests/test_qmutbench_qasm_features.py tests/test_qmutbench_qasm_mutation_diff.py tests/test_qmutbench_corpus_pipeline.py tests/test_qmutbench_master_optimization.py tests/test_qmutbench_learning_curve.py tests/test_qmutbench_selected_pipeline.py tests/test_qmutbench_ml_pipeline_resume.py -q
```
