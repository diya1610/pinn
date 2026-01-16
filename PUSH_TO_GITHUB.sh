#!/bin/bash
# Git initialization and GitHub push script
# Run this to push to GitHub

echo "🚀 PINN Option Pricing - GitHub Push Setup"
echo "=========================================="
echo ""

# Step 1: Initialize Git
echo "Step 1: Initializing Git repository..."
cd /Users/diya/Desktop/proj1
git init
echo "✓ Git initialized"
echo ""

# Step 2: Add all files
echo "Step 2: Adding files (respects .gitignore)..."
git add .
echo "✓ Files staged"
echo ""

# Step 3: Create initial commit
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: PINN option pricing system with Streamlit dashboard

Features:
- Physics-Informed Neural Network for Black-Scholes PDE
- Interactive Streamlit dashboard with real-time pricing
- All 5 Greeks computation via automatic differentiation
- LangChain integration for AI-powered reports
- Pre-trained model checkpoint included
- Comprehensive documentation and examples

Tech Stack:
- PyTorch 2.0+ for neural networks
- Streamlit 1.35.0 for web dashboard
- LangChain + Groq for LLM integration
- NumPy, Pandas, Matplotlib for data analysis"

echo "✓ Initial commit created"
echo ""

# Step 4: Set main branch
echo "Step 4: Setting main branch..."
git branch -M main
echo "✓ Branch renamed to main"
echo ""

# Step 5: Verify files
echo "Step 5: Verifying tracked files..."
echo ""
echo "Files that will be pushed:"
git ls-files | head -20
echo ""
echo "Total files: $(git ls-files | wc -l)"
echo ""

# Step 6: Display next steps
echo "=========================================="
echo "✅ Local repository ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create empty repository on GitHub: https://github.com/new"
echo "   Repository name: pinn-option-pricing"
echo "   Description: Physics-Informed Neural Networks for option pricing"
echo "   Set to PUBLIC for portfolio"
echo ""
echo "2. Run these commands to push:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/pinn-option-pricing.git"
echo "   git push -u origin main"
echo ""
echo "3. Verify on GitHub:"
echo "   https://github.com/YOUR_USERNAME/pinn-option-pricing"
echo ""
echo "=========================================="
echo ""
echo "💡 Tips:"
echo "- Replace YOUR_USERNAME with your actual GitHub username"
echo "- Go to Settings → Add repository topics:"
echo "  pinn, neural-networks, option-pricing, quantitative-finance"
echo "- Add website link to your portfolio (optional)"
echo ""
echo "✨ Your PINN project is ready for internship applications!"
