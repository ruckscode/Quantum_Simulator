$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$runner = Join-Path $root 'experiments\run_classical_source_nested_search.py'
$output = Join-Path $root 'reports\experiments\classical_nested_accuracy_search_20261030'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$log = Join-Path $output 'campaign.log'
$statusPath = Join-Path $output 'campaign_status.json'
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class ClassicalCampaignPower {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
'@
[void][ClassicalCampaignPower]::SetThreadExecutionState([uint32]2147483649)
$tasks = @(
  @{ Name='bugs4q_core'; Input=$null },
  @{ Name='bugs4q_na'; Input=$null },
  @{ Name='bugs4q_pooled'; Input=$null },
  @{ Name='qbugs_qasm_v2'; Input=$null },
  @{ Name='quantum_platform_comment_v2'; Input=$null },
  @{ Name='qolumbina_ideal'; Input=$null },
  @{ Name='lintq_context_v2_conflict_excluded'; Input='reports/experiments/qml_source_preprocessing_20261029/classical_nine_source_v2_inputs/lintq_context_v2_conflict_excluded.csv' },
  @{ Name='bugs4q_robust_execution'; Input=$null },
  @{ Name='qsimbench_exploratory'; Input=$null }
)
$completed = @()
foreach ($task in $tasks) {
  $message = "Starting nested accuracy search: $($task.Name)"
  Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
  $args = @($runner, '--task', $task.Name)
  if ($task.Input) { $args += @('--input', $task.Input) }
  & $python @args 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) {
    $status = [ordered]@{ status='failed'; failed_task=$task.Name; completed=$completed; exit_code=$LASTEXITCODE; updated=(Get-Date).ToString('o'); frozen_test_accessed=$false }
    $status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    exit $LASTEXITCODE
  }
  $completed += $task.Name
  $status = [ordered]@{ status='running'; current_task=$null; completed=$completed; updated=(Get-Date).ToString('o'); frozen_test_accessed=$false }
  if ($completed.Count -lt $tasks.Count) { $status.current_task=$tasks[$completed.Count].Name } else { $status.current_task=$null }
  $status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
$status = [ordered]@{ status='complete'; completed=$completed; updated=(Get-Date).ToString('o'); frozen_test_accessed=$false }
$status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
[void][ClassicalCampaignPower]::SetThreadExecutionState([uint32]2147483648)
