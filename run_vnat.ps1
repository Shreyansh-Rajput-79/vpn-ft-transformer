# Full VNAT pipeline for the GPU lab machine.
#
#   .\run_vnat.ps1 -VnatDir D:\VNAT
#
# Each step is skipped when its output already exists, so the script can be
# re-run after an interruption. Delete an output to force that step again.
# Log of everything: results\vnat_run.log

param(
    [Parameter(Mandatory = $true)][string]$VnatDir,
    [int]$TuneTrials = 20,
    [int]$TuneEpochs = 60,   # same as final training, so trials are not cut short
    [string]$Python = ".\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$py = $Python
New-Item -ItemType Directory -Force results | Out-Null
$log = "results\vnat_run.log"

function Step([string]$name, [string]$output, [string[]]$pyArgs) {
    if (Test-Path $output) {
        Write-Host "[skip] $name ($output exists)"
        return
    }
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    "`n=== $name ($(Get-Date)) ===" | Add-Content $log
    # Windows PowerShell 5.1 turns every stderr line of a native program into an
    # error record; with "Stop" a mere Python warning would abort the run. Judge
    # success by the exit code instead.
    $ErrorActionPreference = "Continue"
    & $py -u @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "Step failed: $name (see $log)" }
}

& $py -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 1. Features: ISCX-style (comparable with ISCX 2016) and NFStream (richer)
Step "Build VNAT features (ISCX extractor)" "data\vnat_features.csv" `
    @("build_vnat_dataset.py", $VnatDir, "-o", "data\vnat_features.csv")
Step "Build VNAT features (NFStream extractor)" "data\vnat_nfstream.csv" `
    @("build_vnat_dataset.py", $VnatDir, "-o", "data\vnat_nfstream.csv", "--extractor", "nfstream")

# 2. Tune the FT-Transformer (validation split only) on the 5-category task.
# VPN vs Non-VPN on NFStream features is near-trivial (minimum packet size alone
# separates it: OpenVPN headers make every tunnelled packet large), so every
# trial scores ~100% there and tuning cannot tell configurations apart.
Step "Tune FT-Transformer (category task)" "results\tuning\best_config.json" `
    @("tune.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--label-col", "category", "--trials", "$TuneTrials", "--epochs", "$TuneEpochs",
      "--out", "results\tuning")

# 3. Final models
Step "Train final VPN model (NFStream)" "models\vnat_model.pth" `
    @("train.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--config", "results\tuning\best_config.json", "--model-out", "models\vnat_model.pth")
Step "Train category model (NFStream)" "models\vnat_category_model.pth" `
    @("train.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--label-col", "category", "--config", "results\tuning\best_config.json",
      "--model-out", "models\vnat_category_model.pth")

# 4. Baseline comparison (LR / RF / XGBoost / FT-Transformer, 5 seeds)
Step "Compare models: VPN, NFStream features" "results\vnat\summary.md" `
    @("compare_models.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--config", "results\tuning\best_config.json", "--out", "results\vnat")
Step "Compare models: VPN, ISCX features" "results\vnat_iscx_features\summary.md" `
    @("compare_models.py", "--data", "data\vnat_features.csv", "--group-col", "source_file",
      "--config", "results\tuning\best_config.json", "--out", "results\vnat_iscx_features")
Step "Compare models: category (multi-class)" "results\vnat_category\summary.md" `
    @("compare_models.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--label-col", "category", "--config", "results\tuning\best_config.json",
      "--out", "results\vnat_category")

# 5. Explainability and security on the final VPN model
Step "Explainability" "results\vnat\explain\summary.md" `
    @("explain.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--model", "models\vnat_model.pth", "--out", "results\vnat\explain")
Step "Security evaluation" "results\vnat\security\summary.md" `
    @("security_eval.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--out", "results\vnat\security")

# 6. Cross-dataset generalization (needs the ISCX feature set on both sides)
Step "Cross-dataset ISCX -> VNAT" "results\cross_dataset_iscx_to_vnat\summary.md" `
    @("cross_dataset.py", "--train", "data\vpn_dataset_cleaned.csv", "--test", "data\vnat_features.csv",
      "--out", "results\cross_dataset_iscx_to_vnat")
Step "Cross-dataset VNAT -> ISCX" "results\cross_dataset_vnat_to_iscx\summary.md" `
    @("cross_dataset.py", "--train", "data\vnat_features.csv", "--test", "data\vpn_dataset_cleaned.csv",
      "--out", "results\cross_dataset_vnat_to_iscx")

# 7. VPN model without TCP-flag counts. Every VNAT VPN flow is OpenVPN over UDP
# (no TCP flags), so the full NFStream model can learn "TCP flags => Non-VPN" and
# miss any VPN that runs over TCP. This model is the one to use on live traffic.
Step "Train VPN model (NFStream, no TCP flags)" "models\vnat_noflags_model.pth" `
    @("train.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--feature-set", "nfstream_noflags", "--config", "results\tuning\best_config.json",
      "--model-out", "models\vnat_noflags_model.pth")
Step "Compare models: VPN, NFStream without TCP flags" "results\vnat_noflags\summary.md" `
    @("compare_models.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--feature-set", "nfstream_noflags", "--config", "results\tuning\best_config.json",
      "--out", "results\vnat_noflags")
Step "Explainability (no TCP flags)" "results\vnat_noflags\explain\summary.md" `
    @("explain.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--model", "models\vnat_noflags_model.pth", "--out", "results\vnat_noflags\explain")

# 8. Explanations and attacks on the non-trivial tasks. VPN vs Non-VPN on NFStream
# features is saturated (~100% for every model, redundant size features), which
# makes the deletion test uninformative; these tasks are not.
Step "Explainability: category model" "results\vnat_category\explain\summary.md" `
    @("explain.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--model", "models\vnat_category_model.pth", "--out", "results\vnat_category\explain")
Step "Train VPN model (timing features only)" "models\vnat_timing_model.pth" `
    @("train.py", "--data", "data\vnat_features.csv", "--group-col", "source_file",
      "--config", "results\tuning\best_config.json", "--model-out", "models\vnat_timing_model.pth")
Step "Explainability: timing-only VPN model" "results\vnat_iscx_features\explain\summary.md" `
    @("explain.py", "--data", "data\vnat_features.csv", "--group-col", "source_file",
      "--model", "models\vnat_timing_model.pth", "--out", "results\vnat_iscx_features\explain")
Step "Security evaluation: timing-only features" "results\vnat_iscx_features\security\summary.md" `
    @("security_eval.py", "--data", "data\vnat_features.csv", "--group-col", "source_file",
      "--out", "results\vnat_iscx_features\security")

Write-Host "`nAll steps done. Results are in results\ (summary.md in each folder)." -ForegroundColor Green
