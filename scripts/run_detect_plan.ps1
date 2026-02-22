param(
    [string]$Plan = "configs/research/stage1_sanity.json",
    [string]$DataRoot = "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset",
    [string]$DataYaml = "data.yaml",
    [string]$TrainSplit = "train",
    [string]$ValSplit = "valid",
    [string]$Device = "cuda",
    [int]$NumWorkers = 4,
    [string]$ClipInit = "",
    [string]$PythonExe = "python",
    [switch]$DryRun,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $Plan)) {
    throw "Plan file not found: $Plan"
}

$planObject = Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json
if ($null -eq $planObject.experiments -or $planObject.experiments.Count -eq 0) {
    throw "Plan has no experiments: $Plan"
}

$total = [int]$planObject.experiments.Count
Write-Host "Loaded plan: $Plan ($total experiment(s))"
if ($planObject.description) {
    Write-Host "Description: $($planObject.description)"
}

$index = 0
foreach ($exp in $planObject.experiments) {
    $index += 1
    $name = [string]$exp.name
    if ([string]::IsNullOrWhiteSpace($name)) {
        $name = "experiment_$index"
    }

    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $index, $total, $name)
    if ($exp.description) {
        Write-Host ("  " + [string]$exp.description)
    }

    $cmd = @(
        $PythonExe,
        "clipdetr/train_detect.py",
        "--data-root", $DataRoot,
        "--data-yaml", $DataYaml,
        "--train-split", $TrainSplit,
        "--val-split", $ValSplit,
        "--device", $Device,
        "--num-workers", [string]$NumWorkers
    )

    if ($exp.use_clip_init -eq $true) {
        if ([string]::IsNullOrWhiteSpace($ClipInit)) {
            Write-Warning "Skipping '$name': use_clip_init=true but -ClipInit was not provided."
            continue
        }
        if (-not (Test-Path -LiteralPath $ClipInit)) {
            Write-Warning "Skipping '$name': clip init checkpoint not found at '$ClipInit'."
            continue
        }
        $cmd += @("--clip-init", $ClipInit)
    }

    if ($exp.cli_args) {
        foreach ($arg in $exp.cli_args) {
            $cmd += [string]$arg
        }
    }

    Write-Host ($cmd -join " ")
    if ($DryRun) {
        continue
    }

    & $cmd[0] @($cmd[1..($cmd.Count - 1)])
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $msg = "Run '$name' failed with exit code $exitCode."
        if ($ContinueOnError) {
            Write-Warning $msg
            continue
        }
        throw $msg
    }
}

Write-Host ""
Write-Host "Plan execution complete."
