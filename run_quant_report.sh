#!/bin/bash
# Quick start script for PINN Quantitative Report Generator

echo "=========================================="
echo "PINN Quantitative Report Generator"
echo "=========================================="

# Check Python environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

# Activate environment
source .venv/bin/activate

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; import langchain_groq; print('✅ All dependencies found')" 2>/dev/null || {
    echo "⚠️  Installing missing dependencies..."
    pip install -r requirements.txt
}

# Check model file
if [ ! -f "pinn_bs_best.pth" ]; then
    echo "❌ Model file 'pinn_bs_best.pth' not found!"
    echo "Please run pinn.ipynb first to train and save the model"
    exit 1
fi

echo "✅ All checks passed!"
echo ""
echo "Running Quantitative Report Generator..."
echo "=========================================="
echo ""

# Run the generator
python quant_report_generator.py

echo ""
echo "=========================================="
echo "✅ Report generation completed!"
echo "=========================================="
