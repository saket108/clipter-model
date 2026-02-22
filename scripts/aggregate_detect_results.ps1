param(
    [string]$ExperimentsRoot = "experiments",
    [string]$OutputCsv = "reports/detect_runs_flat.csv",
    [string]$OutputGroupedCsv = "reports/detect_runs_grouped.csv",
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

& $PythonExe "clipdetr/utils/aggregate_detect_results.py" `
    --experiments-root $ExperimentsRoot `
    --output-csv $OutputCsv `
    --output-grouped-csv $OutputGroupedCsv
