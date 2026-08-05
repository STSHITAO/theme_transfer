param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$ControllerPid = 7292,
    [int]$IntervalSeconds = 600,
    [int]$StaleThresholdSeconds = 1200,
    [switch]$AutoRestart
)

$ErrorActionPreference = 'Continue'
$pythonPath = 'D:\Anaconda\envs\pytorch\python.exe'
$logDir = Join-Path $Root 'data\packages\_full_generation_logs'
$statusPath = Join-Path $logDir 'monitor_status.jsonl'
$summaryPath = Join-Path $Root 'data\evaluations\eval_full_structure_v1_summary.json'
$sequenceScript = Join-Path $Root 'scripts\run_full_pipeline_sequence.ps1'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-ThemeProgress {
    param([string]$ThemeId)
    $path = Join-Path $logDir "$ThemeId.stdout.log"
    if (-not (Test-Path $path)) { return 0 }
    return @(Select-String -Path $path -Pattern '"event": "case_complete"' -ErrorAction SilentlyContinue).Count
}

function Get-LatestLogTime {
    $files = @(Get-ChildItem $logDir -Filter '*.log' -File -ErrorAction SilentlyContinue)
    if (-not $files) { return $null }
    return ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
}

function Write-Status {
    param([hashtable]$Status)
    $Status.timestamp = (Get-Date).ToString('o')
    Add-Content -LiteralPath $statusPath -Value (($Status | ConvertTo-Json -Compress)) -Encoding UTF8
}

while ($true) {
    if (Test-Path $summaryPath) {
        Write-Status @{ event = 'monitor_complete'; reason = 'evaluation_summary_exists' }
        break
    }

    $pythonProcesses = @(Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $pythonPath })
    $controller = Get-Process -Id $ControllerPid -ErrorAction SilentlyContinue
    $latestLog = Get-LatestLogTime
    $staleSeconds = if ($latestLog) { [int]((Get-Date) - $latestLog).TotalSeconds } else { -1 }
    $progress = @{
        theme_001 = Get-ThemeProgress 'theme_001'
        theme_002 = Get-ThemeProgress 'theme_002'
        theme_003 = Get-ThemeProgress 'theme_003'
        theme_004 = Get-ThemeProgress 'theme_004'
    }
    $stderrBytes = @(Get-ChildItem $logDir -Filter '*.stderr.log' -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum

    if ($pythonProcesses.Count -gt 0 -or $controller) {
        Write-Status @{
            event = 'monitor_tick'
            state = if ($pythonProcesses.Count -gt 0) { 'running' } else { 'controller_waiting' }
            controller_pid = $ControllerPid
            python_pids = @($pythonProcesses | Select-Object -ExpandProperty Id)
            progress = $progress
            stale_seconds = $staleSeconds
            stderr_bytes = [int64]$stderrBytes
        }
    } elseif (($staleSeconds -ge $StaleThresholdSeconds -or $staleSeconds -lt 0) -and $AutoRestart) {
        $restartStamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
        $stdout = Join-Path $logDir "monitor_restart_$restartStamp.stdout.log"
        $stderr = Join-Path $logDir "monitor_restart_$restartStamp.stderr.log"
        Write-Status @{
            event = 'pipeline_interrupted'
            reason = 'controller_and_python_absent_with_stale_logs'
            previous_controller_pid = $ControllerPid
            progress = $progress
            stale_seconds = $staleSeconds
            stderr_bytes = [int64]$stderrBytes
            restart_stdout = $stdout
            restart_stderr = $stderr
        }
        $newController = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $sequenceScript, '-Root', $Root) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $ControllerPid = $newController.Id
    } elseif ($staleSeconds -ge $StaleThresholdSeconds -or $staleSeconds -lt 0) {
        Write-Status @{
            event = 'pipeline_interrupted'
            reason = 'controller_and_python_absent_with_stale_logs'
            action = 'monitor_only_waiting_for_user_authorization'
            previous_controller_pid = $ControllerPid
            progress = $progress
            stale_seconds = $staleSeconds
            stderr_bytes = [int64]$stderrBytes
        }
    } else {
        Write-Status @{
            event = 'monitor_tick'
            state = 'no_python_but_controller_absent_recent_log'
            controller_pid = $ControllerPid
            progress = $progress
            stale_seconds = $staleSeconds
            stderr_bytes = [int64]$stderrBytes
        }
    }

    Start-Sleep -Seconds $IntervalSeconds
}
