# 🚀 QUICK START GUIDE

## One-Minute Setup

### Step 1: Navigate to Project
```bash
cd /Users/diya/Desktop/proj1
```

### Step 2: Start the Dashboard
```bash
# Option A: Using the launcher script
chmod +x run_streamlit.sh
./run_streamlit.sh

# Option B: Direct command
/Users/diya/Desktop/proj1/.venv/bin/streamlit run streamlit_app.py
```

### Step 3: Open in Browser
```
http://localhost:8501
```

## Dashboard Quick Reference

### 🎛️ Sidebar Controls
```
Spot Price (S)      → Adjust with slider (1-150)
Strike Price (K)    → Adjust with slider (10-150)
Time (t)            → Adjust with slider (0-1 years)
Rate (r)            → Adjust with slider (0-10%)
Volatility (σ)      → Adjust with slider (5-100%)

[🔄 Load Model]     → Load PINN model
[📈 Generate Report] → Create AI analysis
```

### 5 Main Tabs

| Tab | Purpose |
|-----|---------|
| **Pricing** | Compare PINN vs Black-Scholes prices |
| **Greeks** | View all 5 Greeks with curves |
| **Sensitivity** | See impact of parameter changes |
| **Comparison** | 2D surface analysis |
| **Report** | Generate AI-powered analysis |

## Example Workflows

### Workflow 1: Quick Price Check
1. Load model
2. Adjust S, K, t, r, σ
3. See instant price in Pricing tab
4. Check pricing curve

### Workflow 2: Greeks Analysis
1. Load model
2. Set parameters
3. Go to Greeks tab
4. See all 5 Greeks with curves
5. Check analytical comparison

### Workflow 3: Risk Analysis
1. Load model
2. Set parameters
3. Sensitivity tab → Analyze spot price impact
4. Greeks tab → Check Delta/Gamma
5. Report tab → Generate full analysis

### Workflow 4: Full Report Generation
1. Load model
2. Set option parameters
3. Go to Report tab
4. Click "Generate AI Report"
5. Read professional analysis
6. Download report

## Common Parameters

### ATM (At-The-Money)
- Spot = Strike: S=K=50
- t=0.25, r=5%, σ=25%
- Good for Greeks visualization

### OTM (Out-of-The-Money)
- Spot < Strike: S=40, K=50
- t=0.25, r=5%, σ=25%
- Lower option value

### ITM (In-The-Money)
- Spot > Strike: S=60, K=50
- t=0.25, r=5%, σ=25%
- Higher option value, lower theta decay

### Near Expiry
- t → 0.95 (close to maturity)
- Shows theta decay effects
- Greeks become more sensitive

## Tips & Tricks

✅ **Do:**
- Start with ATM options (S≈K)
- Use Sensitivity tab to understand Greeks
- Generate report for complete analysis
- Try different volatility values

❌ **Don't:**
- Set S or K to extreme values
- Go too close to expiration (t>0.99)
- Adjust too many parameters at once
- Ignore the error percentages

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `r` | Rerun app |
| `c` | Clear cache |
| `Ctrl+C` | Stop server |

## Browser Tips

- **Best**: Chrome, Safari, Firefox
- **Mobile**: Not optimized for small screens
- **Display**: Use full-screen for best experience
- **Performance**: Hard refresh if plots don't update

## File Outputs

When you use the Report tab:
```
📁 reports_YYYYMMDD_HHMMSS/
├── report.html       (Visual report)
├── comparisons.png   (Price comparison)
└── price_surface.png (2D surface plot)
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "Model won't load" | Click "Load Model" again |
| "API Error" | Check GROQ_API_KEY in .env |
| "Plots not showing" | Refresh browser (F5) |
| "Slow response" | Check internet connection |
| "Out of memory" | Restart Streamlit app |

## Need Help?

### Check Documentation
```bash
less STREAMLIT_README.md
```

### View Model Details
```bash
jupyter notebook pinn.ipynb
```

### Run Backend Generator
```bash
python quant_report_generator.py
```

## Next Steps

After running the dashboard:

1. **Explore Different Scenarios**
   - Try ATM, OTM, ITM options
   - Vary volatility
   - Change time to expiration

2. **Understand Greeks**
   - See how each Greek changes
   - Compare PINN vs Analytical
   - Use sensitivity analysis

3. **Generate Reports**
   - Create comprehensive analysis
   - Download for documentation
   - Share with team

4. **Integrate with Workflows**
   - Use API from Python
   - Embed in other applications
   - Build custom analyses

---

**Ready?** Start with Step 1 above! 🚀

**Questions?** Check STREAMLIT_README.md for detailed guide.
