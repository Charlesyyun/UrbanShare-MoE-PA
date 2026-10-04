param(
    [string]$OutputRoot = 'baselines/outputs/main',
    [int]$Seed = 1234,
    [int]$Epochs = 50,
    [int]$MaxAgents = 0
)
$ErrorActionPreference = 'Stop'
$BaselinePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $BaselinePython)) { throw 'Create baselines/.venv first; see README.' }
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    foreach ($BaselineModel in @('persistence', 'mean7', 'catboost', 'gru', 'lstm', 'transformer', 'tabm', 'timexer', 'logistic_normal', 'mdcev')) {
        $BaselineArgs = @('-m', 'baselines.run', '--model', $BaselineModel,
            '--out', "$OutputRoot/$BaselineModel-seed$Seed", '--seed', "$Seed", '--epochs', "$Epochs",
            '--history', '7', '--split-mode', 'phase_last_week_test')
        if ($BaselineModel -eq 'catboost') { $BaselineArgs += @('--depth','6') }
        if ($BaselineModel -notin @('persistence', 'mean7')) { $BaselineArgs += '--include-agent-id' }
        if ($MaxAgents -gt 0) { $BaselineArgs += @('--max-agents', "$MaxAgents") }
        & $BaselinePython @BaselineArgs
        if ($LASTEXITCODE -ne 0) { throw "$BaselineModel failed with exit code $LASTEXITCODE" }
    }
} finally { Pop-Location }
