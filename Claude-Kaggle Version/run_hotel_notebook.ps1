# Hotel Cancellation Prediction - Setup & Launch
Write-Host "Setting up environment..." -ForegroundColor Cyan

# Find the Downloads folder
$DownloadsFolder = "$env:USERPROFILE\Downloads"
Set-Location $DownloadsFolder

# Extract the dataset zip if needed
$zipFile = "$DownloadsFolder\hotel-booking-demand.zip"
if (Test-Path $zipFile) {
    Write-Host "Extracting dataset..." -ForegroundColor Yellow
    Expand-Archive -Path $zipFile -DestinationPath $DownloadsFolder -Force
    Write-Host "Dataset extracted!" -ForegroundColor Green
}

# Install required Python packages
Write-Host "Installing Python packages (this may take a few minutes)..." -ForegroundColor Yellow
pip install imbalanced-learn xgboost lightgbm catboost notebook ipykernel -q

# Launch Jupyter and open the notebook
Write-Host "Launching Jupyter Notebook..." -ForegroundColor Green
jupyter notebook "$DownloadsFolder\hotel_cancellation_prediction.ipynb"