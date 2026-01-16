"""
Example Scenarios for PINN Quantitative Report Generator

This file contains pre-configured scenarios you can analyze without typing interactively.
Modify and use these as templates for different analysis cases.
"""

EXAMPLE_SCENARIOS = {
    "at_the_money": {
        "S": 50.0,          # Spot = Strike (ATM)
        "K": 50.0,
        "t": 0.25,          # 3 months to expiry
        "r": 0.05,
        "sigma": 0.25,
        "T": 1.0,
        "description": "At-the-money option, 3 months to expiry"
    },
    
    "out_of_the_money": {
        "S": 40.0,          # Spot < Strike (OTM)
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.25,
        "T": 1.0,
        "description": "Out-of-the-money option, 3 months to expiry"
    },
    
    "in_the_money": {
        "S": 60.0,          # Spot > Strike (ITM)
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.25,
        "T": 1.0,
        "description": "In-the-money option, 3 months to expiry"
    },
    
    "deep_itm": {
        "S": 100.0,         # Deep in the money
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.25,
        "T": 1.0,
        "description": "Deep in-the-money option"
    },
    
    "near_expiry": {
        "S": 50.0,
        "K": 50.0,
        "t": 0.95,          # Near expiry (5 days remaining)
        "r": 0.05,
        "sigma": 0.25,
        "T": 1.0,
        "description": "At-the-money option near expiry"
    },
    
    "high_volatility": {
        "S": 50.0,
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.50,      # High volatility
        "T": 1.0,
        "description": "High volatility environment (σ=50%)"
    },
    
    "low_volatility": {
        "S": 50.0,
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.10,      # Low volatility
        "T": 1.0,
        "description": "Low volatility environment (σ=10%)"
    },
    
    "high_rates": {
        "S": 50.0,
        "K": 50.0,
        "t": 0.25,
        "r": 0.10,          # High risk-free rate
        "sigma": 0.25,
        "T": 1.0,
        "description": "High interest rate environment"
    },
    
    "low_rates": {
        "S": 50.0,
        "K": 50.0,
        "t": 0.25,
        "r": 0.01,          # Low risk-free rate
        "sigma": 0.25,
        "T": 1.0,
        "description": "Low interest rate environment"
    },
    
    "market_crash": {
        "S": 30.0,          # Sharp drop
        "K": 50.0,
        "t": 0.25,
        "r": 0.05,
        "sigma": 0.50,      # Increased volatility
        "T": 1.0,
        "description": "Market crash scenario: spot drop + vol spike"
    }
}

# ==================== PROGRAMMATIC USAGE ====================
"""
To use these scenarios programmatically instead of interactive input:

from quant_report_generator import QuantReportGenerator

# Create generator
gen = QuantReportGenerator()

# Run with predefined scenario
scenario = EXAMPLE_SCENARIOS["at_the_money"]
results = gen.price_option(scenario)

# Or loop through multiple scenarios
for scenario_name, scenario_params in EXAMPLE_SCENARIOS.items():
    print(f"Analyzing: {scenario_params['description']}")
    results = gen.price_option(scenario_params)
    # Generate report, plots, etc.
"""

if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("PINN Quantitative Report Generator - Example Scenarios")
    print("=" * 80)
    
    for name, scenario in EXAMPLE_SCENARIOS.items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"Description: {scenario['description']}")
        print(f"Parameters:")
        print(f"  Spot Price (S):        ${scenario['S']:.2f}")
        print(f"  Strike Price (K):      ${scenario['K']:.2f}")
        print(f"  Time (t):              {scenario['t']:.2f} years ({(1-scenario['t'])*365:.0f} days to expiry)")
        print(f"  Risk-free Rate (r):    {scenario['r']*100:.2f}%")
        print(f"  Volatility (σ):        {scenario['sigma']*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("To use a scenario programmatically:")
    print("  from example_scenarios import EXAMPLE_SCENARIOS")
    print("  scenario = EXAMPLE_SCENARIOS['at_the_money']")
    print("=" * 80)
