param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$python = 'D:\Anaconda\envs\pytorch\python.exe'
Set-Location $Root
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TPQS_DEVICE = 'cuda:0'
$env:TPQS_BATCH_SIZE = '2'

& $python 'scripts/run_full_generation.py' '--theme-id' 'theme_001' '--theme-id' 'theme_002' '--theme-id' 'theme_003' '--theme-id' 'theme_004' '--candidate-count' '2'
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python 'scripts/evaluate_full_packages.py'
exit $LASTEXITCODE
