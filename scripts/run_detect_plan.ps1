param(
    [string]$Plan = "configs/research/stage1_sanity.json",
    [string]$DataRoot = "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset",
    [string]$DataYaml = "data.yaml",
    [string]$TrainSplit = "train",
    [string]$ValSplit = "valid",
    [string]$Device = "auto",
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

$resolvedDevice = $Device
if ($Device -eq "cuda") {
    try {
        $cudaCheck = (& $PythonExe -c "import torch; print(1 if torch.cuda.is_available() else 0)")
        $cudaCheck = ($cudaCheck | Select-Object -Last 1).ToString().Trim()
        if ($cudaCheck -ne "1") {
            Write-Warning "Requested -Device cuda, but CUDA is not available in this Python environment. Falling back to CPU."
            $resolvedDevice = "cpu"
        }
    } catch {
        Write-Warning "Could not verify CUDA availability. Keeping -Device cuda."
    }
}

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
        "--device", $resolvedDevice,
        "--num-workers", [string]$NumWorkers
    )

    if ($exp.use_clip_init -eq $true) {
        if (-not [string]::IsNullOrWhiteSpace($ClipInit) -and (Test-Path -LiteralPath $ClipInit)) {
            $cmd += @("--clip-init", $ClipInit)
        }
        else {
            if ([string]::IsNullOrWhiteSpace($ClipInit)) {
                Write-Warning "Plan item '$name' requested CLIP init but -ClipInit was not provided. Falling back to auto CLIP-init discovery."
            }
            else {
                Write-Warning "Plan item '$name' requested CLIP init but checkpoint not found at '$ClipInit'. Falling back to auto CLIP-init discovery."
            }
            $cmd += @("--auto-clip-init")
        }
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
