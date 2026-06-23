$ErrorActionPreference = "Stop"

$envName = "rlgame-ppo"
$envExists = conda env list | Select-String -Pattern "^\s*$envName\s"

if ($envExists) {
    Write-Host "Conda environment '$envName' already exists. Updating from environment.yml..."
    conda env update -n $envName -f environment.yml --prune
} else {
    Write-Host "Creating conda environment '$envName' from environment.yml..."
    conda env create -f environment.yml
}

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Activate it with: conda activate $envName"
Write-Host "Smoke train: python main.py --mode train --max-train-steps 2 --batch-size 4 --mini-batch-size 2 --max-episode-steps 4 --no-plot"
Write-Host "Smoke test:  python main.py --mode test --max-episode-steps 4 --no-plot"
