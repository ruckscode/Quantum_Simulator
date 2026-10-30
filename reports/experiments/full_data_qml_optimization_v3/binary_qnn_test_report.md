# Binary QNN trainer tests

The focused integrity suite covers label/split checks, deterministic sampling, fold checkpointing, epoch checkpoint recovery, atomic-write interruption, configuration and fingerprint validation, invalid checkpoint rejection, and memory-gate behavior. These tests use synthetic fixtures and mocks; they do not execute a quantum circuit or establish a real gradient/update.

Latest command (pytest plugin autoload disabled; workspace-local temporary paths):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TEMP=Join-Path (Get-Location).Path '.pytest-codex-binary-temp'
$env:TMP=$env:TEMP
$base=Join-Path (Get-Location).Path '.pytest-codex-binary-base-2'
.venv\Scripts\python.exe -m pytest --basetemp=$base tests\test_qmutbench_binary_qnn.py -q
```

Latest result: **20 passed in 0.69s**. Available RAM was 1,224 MiB immediately before the test run and 1,238 MiB after it (7,488 MiB total). The tests do not import PennyLane or start training. The required campaign startup threshold remains 2,048 MiB, so no campaign was launched.

Two regression cases were added after the earlier 18-test result:

- Epoch restore rejects the wrong fold, a different data split fingerprint, or changed configuration.
- An epoch with a loss history shorter than its claimed completed epoch is not written or reported as the latest completed boundary; the preceding valid epoch remains available.
