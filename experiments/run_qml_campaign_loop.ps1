$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
$output = Join-Path $root 'reports\experiments\qmutbench_binary_qnn_full_development_chunk16_20261029'
$campaign = Join-Path $root 'reports\experiments\qml_chunked_execution_campaign_20261030'
$runtime = Join-Path $campaign 'runtime'
$log = Join-Path $runtime 'continuous_resume.log'
$statusFile = Join-Path $runtime 'continuous_resume_status.json'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class QmlCampaignPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
[void][QmlCampaignPower]::SetThreadExecutionState([uint32]2147483649)

function Write-RunStatus([string]$status, [string]$message, [int]$availableMiB) {
    $record = [ordered]@{
        updated_local = (Get-Date).ToString('o')
        status = $status
        message = $message
        available_ram_mib = $availableMiB
        checkpoint = 'latest verified fold checkpoint is recorded in full_campaign_manifest.json'
        chunk_batches = 16
        max_chunks_per_invocation = 128
        frozen_test_accessed = $false
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusFile -Encoding UTF8
}

function Get-AvailableRamMiB {
    $value = & $python -c "import psutil; print(int(psutil.virtual_memory().available / 1048576))"
    return [int]($value | Select-Object -First 1)
}

while ($true) {
    $availableMiB = Get-AvailableRamMiB
    if ($availableMiB -lt 2112) {
        $message = "Waiting for memory preflight: $availableMiB MiB available; need 2112 MiB to start safely."
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
        Write-RunStatus 'waiting_for_memory' $message $availableMiB
        Start-Sleep -Seconds 300
        continue
    }

    $start = Get-Date
    Add-Content -LiteralPath $log -Value "[$($start.ToString('o'))] Starting up to 128 checkpointed chunks; available RAM ${availableMiB} MiB."
    $runLog = Join-Path $runtime ("runner_{0}.log" -f $start.ToString('yyyyMMdd_HHmmss'))
    & $python -m qmutbench_qml.training.run_binary_development `
        --mode full-development `
        --output $output `
        --chunk-batches 16 `
        --max-chunks-per-invocation 128 `
        --resume 2>&1 | Tee-Object -FilePath $runLog
    $exitCode = $LASTEXITCODE

    $manifestPath = Join-Path $output 'full_campaign_manifest.json'
    $manifest = if (Test-Path -LiteralPath $manifestPath) {
        Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    } else { $null }
    if ($exitCode -eq 3) {
        $message = 'Runner memory preflight declined to start; waiting before retrying from the last verified checkpoint.'
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
        Write-RunStatus 'waiting_for_memory' $message (Get-AvailableRamMiB)
        Start-Sleep -Seconds 300
        continue
    }
    if ($manifest -and $manifest.status -eq 'completed_full_development_oof' -and @($manifest.completed_folds).Count -eq 5) {
        $message = 'All five development folds completed with OOF outputs.'
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
        Write-RunStatus 'completed' $message (Get-AvailableRamMiB)
        break
    }

    if ($exitCode -eq 0 -or ($manifest -and $manifest.status -eq 'stopped_memory_guard')) {
        $completed = if ($manifest) { @($manifest.completed_folds).Count } else { 0 }
        $committed = if ($manifest) { [int]$manifest.total_chunks_committed } else { 0 }
        $message = "Checkpointed continuation; completed folds=$completed; committed chunks=$committed; exit=$exitCode."
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
        Write-RunStatus 'checkpointed_continuation' $message (Get-AvailableRamMiB)
        continue
    }

    $message = "Stopped on a non-memory error (exit $exitCode). Check continuous_resume.log."
    Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] $message"
    Write-RunStatus 'stopped_error' $message (Get-AvailableRamMiB)
    break
}
