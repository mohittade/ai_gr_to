# Installation script for the German-English-Marathi Translation System
# Run this script to set up your environment

Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "German → English → Marathi Translation System - Setup"
Write-Host "=" -NoNewline; 1..79 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""

# Check Python version
Write-Host "`n[1] Checking Python version..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n[2] Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "⚠ Virtual environment already exists, skipping creation" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[3] Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "`n[4] Upgrading pip..."
python -m pip install --upgrade pip
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install PyTorch
Write-Host "`n[5] Installing PyTorch..."
Write-Host "   Checking for CUDA availability..."

# Check if CUDA is available
$cudaAvailable = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ CUDA detected, installing PyTorch with CUDA support" -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "   ⚠ CUDA not detected, installing CPU-only PyTorch" -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

# Install other requirements
Write-Host "`n[6] Installing other dependencies..."
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create necessary directories
Write-Host "`n[7] Creating project directories..."
$directories = @(
    "data/german_english",
    "data/english_marathi",
    "checkpoints",
    "logs",
    "evaluation_results"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "   ✓ Created $dir" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ $dir already exists" -ForegroundColor Yellow
    }
}

# Download NLTK data
Write-Host "`n[8] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
Write-Host "✓ NLTK data downloaded" -ForegroundColor Green

# Verify installation
Write-Host "`n[9] Verifying installation..."
Write-Host "   Checking PyTorch..."
$torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ $torchCheck" -ForegroundColor Green
    
    # Check CUDA
    $cudaCheck = python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1
    Write-Host "   ✓ $cudaCheck" -ForegroundColor Green
} else {
    Write-Host "   ✗ PyTorch installation failed" -ForegroundColor Red
}

Write-Host "   Checking transformers..."
$transformersCheck = python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ $transformersCheck" -ForegroundColor Green
} else {
    Write-Host "   ✗ Transformers installation failed" -ForegroundColor Red
}

Write-Host "   Checking FastAPI..."
$fastapiCheck = python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ $fastapiCheck" -ForegroundColor Green
} else {
    Write-Host "   ✗ FastAPI installation failed" -ForegroundColor Red
}

# Summary
Write-Host "`n" + ("=" * 80)
Write-Host "✓ Installation complete!" -ForegroundColor Green
Write-Host ("=" * 80)

Write-Host "`nNext steps:"
Write-Host "1. Prepare your datasets in the data/ directory"
Write-Host "2. Train models: python examples/train_example.py"
Write-Host "3. Evaluate models: python examples/evaluate_example.py"
Write-Host "4. Start API server: python api/api_server.py"
Write-Host "5. Test translations: python examples/basic_translation.py"

Write-Host "`nFor more information, see README.md"
Write-Host ""
