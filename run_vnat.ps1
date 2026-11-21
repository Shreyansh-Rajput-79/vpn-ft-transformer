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
    & $py -u @pyArgs 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (see $log)" }
}

& $py -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 1. Features: ISCX-style (comparable with ISCX 2016) and NFStream (richer)
Step "Build VNAT features (ISCX extractor)" "data\vnat_features.csv" `
    @("build_vnat_dataset.py", $VnatDir, "-o", "data\vnat_features.csv")
Step "Build VNAT features (NFStream extractor)" "data\vnat_nfstream.csv" `
    @("build_vnat_dataset.py", $VnatDir, "-o", "data\vnat_nfstream.csv", "--extractor", "nfstream")

# 2. Tune the FT-Transformer (validation split only)
Step "Tune FT-Transformer" "results\tuning\best_config.json" `
    @("tune.py", "--data", "data\vnat_nfstream.csv", "--group-col", "source_file",
      "--trials", "$TuneTrials", "--out", "results\tuning")

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

Write-Host "`nAll steps done. Results are in results\ (summary.md in each folder)." -ForegroundColor Green
