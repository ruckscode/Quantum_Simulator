# Binary QNN trainer tests

Command (workspace Temp redirection avoids protected Windows Temp):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:TEMP = Join-Path (Get-Location) '.tmp-binary-qnn-tests'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pytest --basetemp="$PWD\.pytest-binary-qnn" tests\test_qmutbench_binary_qnn.py -q
```

Result: **18 passed**. Tests include mocked fold training, atomic fold and per-epoch checkpoint integrity, resume after interruption at a completed epoch boundary, stale/corrupt checkpoint rejection, startup and in-run RAM gates, and a source-level inference-call regression check. No QML circuit was imported or fitted. The latest guarded campaign invocation stopped before scientific imports because available RAM was 649 MiB against the 2,048 MiB startup minimum.
