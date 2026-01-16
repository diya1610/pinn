# PINN Option Pricing - Quick Start Guide

Get up and running in 5 minutes.

## Installation (2 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/pinn-option-pricing.git
cd pinn-option-pricing

# 2. Create environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key (optional for reports)
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com
```

## Run Dashboard (1 minute)

```bash
streamlit run streamlit_app.py
```

Opens at: **http://localhost:8501**

## First-Time Setup

1. **Click "🔄 Load Model"** - Loads pre-trained PINN checkpoint (2.8MB, takes 2-3 seconds)
2. **Verify loading** - Should show green ✓ when complete
3. **Ready to use** - All 5 tabs now functional

## Tab Guide

| Tab | Purpose | Key Action |
|-----|---------|-----------|
| **Pricing** | Real-time option prices | Adjust S (spot), t (time), K (strike), r (rate), σ (vol) sliders |
| **Greeks** | All 5 Greeks (Δ,Γ,ν,Θ,ρ) | Compare PINN vs Black-Scholes |
| **Sensitivity** | Parameter impact analysis | Choose parameter to vary, view heatmap |
| **Comparison** | 2D surface pricing | See PINN vs analytical across grid |
| **Report** | AI-powered analysis | Click "Generate AI Report" for LLM insights |

## Model Parameters

Best accuracy (0.01% error):
- Strike (K): **$50** ✓
- Rate (r): **5%** ✓
- Volatility (σ): **25%** ✓

⚠️ Deviations reduce accuracy. Warnings appear in sidebar if you deviate.

## Common Tasks

### Check if Model Loaded
Look for green checkmark in sidebar under "Model Status"

### Generate a Report
1. Set parameters (sidebar sliders)
2. Click "🤖 Generate AI Report"
3. Wait 5-10 seconds for LLM analysis
4. Review professional summary

### Export Data
Currently dashboard shows visualizations. For programmatic access, modify `streamlit_app.py` to export DataFrames as CSV.

### Run Reports from CLI
```bash
python quant_report_generator.py
# Follow interactive prompts
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Module not found" | Run `pip install -r requirements.txt` |
| "Model file not found" | Verify `pinn_bs_best.pth` exists in project root |
| "GROQ_API_KEY error" | Copy `.env.example` to `.env` and add your key |
| Port 8501 in use | Run `streamlit run streamlit_app.py --server.port=8502` |
| Slow on CPU | GPU available if CUDA installed; uses GPU automatically |

## Architecture Overview

```
streamlit_app.py (Main Dashboard)
    ├─ Neural Network (ImprovedPINN)
    │   ├─ 2 inputs: S (spot), τ (time-to-maturity)
    │   ├─ 4 hidden layers: 128 neurons each
    │   └─ 1 output: Call option price
    │
    ├─ Greeks Engine
    │   ├─ Automatic differentiation via PyTorch
    │   └─ Computes: Δ, Γ, ν, Θ, ρ
    │
    └─ LangChain Integration
        ├─ Groq LLM for report generation
        └─ Professional quantitative analysis
```

## Next Steps

1. **Explore pricing tab** - Understand how price changes with parameters
2. **Check Greeks tab** - See automatic differentiation in action
3. **Run sensitivity** - Identify which parameters matter most
4. **Generate reports** - See AI-powered analysis
5. **Review code** - Study how PINN pricing works

## Resources

- **PINN Paper**: [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561)
- **Black-Scholes**: [Wikipedia article](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- **PyTorch Autograd**: [Official docs](https://pytorch.org/docs/stable/autograd.html)
- **Streamlit**: [streamlit.io](https://docs.streamlit.io/)

## Support

- Check main `README.md` for full documentation
- Review code comments in `streamlit_app.py`
- See `ACCURACY_EXPLANATION.md` for parameter sensitivity details
- Open a GitHub issue for bugs or feature requests

**Ready? Run `streamlit run streamlit_app.py` and start exploring! 🚀**
