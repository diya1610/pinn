"""
Quantitative Report Generator for PINN Black-Scholes Model
Integrates LangChain for intelligent report generation with user input
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
import json
from io import StringIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== CONFIG ====================
class Config:
    """Model configuration"""
    K = 50.0
    r = 0.05
    sigma = 0.25
    T = 1.0
    S_max = 150.0
    
    layers = [2, 128, 128, 128, 128, 1]
    activation = 'tanh'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'pinn_bs_best.pth'

config = Config()

# ==================== ANALYTICAL SOLUTIONS ====================
def bs_call_price(S: np.ndarray, K: float, r: float, sigma: float, 
                  t: float, T: float) -> np.ndarray:
    """Black-Scholes analytical call option price."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    price = S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    price[S <= 1e-10] = 0.0
    return price

def bs_delta(S: np.ndarray, K: float, r: float, sigma: float, 
             t: float, T: float) -> np.ndarray:
    """Black-Scholes delta"""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    delta = norm.cdf(d1)
    delta[S <= 1e-10] = 0.0
    return delta

# ==================== NORMALIZATION ====================
class Normalizer:
    """Input/output normalization"""
    def __init__(self, S_max: float, T: float):
        self.S_mean = S_max / 2
        self.S_std = S_max / 4
        self.tau_mean = T / 2
        self.tau_std = T / 4
        self.u_mean = S_max / 4
        self.u_std = S_max / 4
    
    def normalize_input(self, X: torch.Tensor) -> torch.Tensor:
        X_norm = X.clone()
        X_norm[:, 0:1] = (X[:, 0:1] - self.S_mean) / self.S_std
        X_norm[:, 1:2] = (X[:, 1:2] - self.tau_mean) / self.tau_std
        return X_norm
    
    def denormalize_output(self, u_norm: torch.Tensor) -> torch.Tensor:
        return u_norm * self.u_std + self.u_mean

# ==================== NEURAL NETWORK ====================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear(x))

class EnhancedPINN(nn.Module):
    """Enhanced PINN with residual connections"""
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        super().__init__()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()
        
        self.layers_list = nn.ModuleList()
        self.layers_list.append(nn.Linear(layers[0], layers[1]))
        
        for i in range(1, len(layers)-2):
            if layers[i] == layers[i+1]:
                self.layers_list.append(ResidualBlock(layers[i], self.activation))
            else:
                self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
        
        self.layers_list.append(nn.Linear(layers[-2], layers[-1]))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for i, layer in enumerate(self.layers_list[:-1]):
            if isinstance(layer, ResidualBlock):
                z = layer(z)
            else:
                z = self.activation(layer(z))
        z = self.layers_list[-1](z)
        return z

# ==================== PINN PRICER ====================
class PINNPricer:
    """Easy-to-use interface for option pricing"""
    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.normalizer = checkpoint['normalizer']
        
        config = Config()
        self.model = EnhancedPINN(config.layers, config.activation)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.config = config
        self.device = config.device
        self.model.to(self.device)
    
    def price(self, S: float, t: float) -> float:
        """Price a call option"""
        tau = self.config.T - t
        X = torch.tensor([[S, tau]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_norm = self.normalizer.normalize_input(X)
            u_norm = self.model(X_norm)
            u = self.normalizer.denormalize_output(u_norm)
        
        return u.item()
    
    def greeks(self, S: float, t: float) -> dict:
        """Compute Greeks"""
        tau = self.config.T - t
        S_tensor = torch.tensor([[S]], dtype=torch.float32, 
                               requires_grad=True).to(self.device)
        tau_tensor = torch.tensor([[tau]], dtype=torch.float32).to(self.device)
        X = torch.cat([S_tensor, tau_tensor], dim=1)
        
        X_norm = self.normalizer.normalize_input(X)
        u_norm = self.model(X_norm)
        u = self.normalizer.denormalize_output(u_norm)
        
        delta = torch.autograd.grad(u, S_tensor, create_graph=True)[0]
        
        gamma = torch.autograd.grad(delta, S_tensor)[0]
        
        return {
            'price': u.item(),
            'delta': delta.item(),
            'gamma': gamma.item()
        }
    
    def price_batch(self, S_array: np.ndarray, t: float) -> np.ndarray:
        """Price multiple spots at once"""
        tau = self.config.T - t
        n = len(S_array)
        X = torch.tensor(
            np.column_stack([S_array, np.full(n, tau)]),
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            X_norm = self.normalizer.normalize_input(X)
            u_norm = self.model(X_norm)
            u = self.normalizer.denormalize_output(u_norm)
        
        return u.cpu().numpy().flatten()

# ==================== REPORT GENERATOR ====================
class QuantReportGenerator:
    """Imagine you are a quant researcher working in a company as qualified as citadel Generate professional quantitative reports with valuable explaination of the results."""
    
    def __init__(self, checkpoint_path: str = 'pinn_bs_best.pth'):
        """Initialize pricer and LLM"""
        self.pricer = PINNPricer(checkpoint_path)
        self.config = Config()
        
        # Initialize LangChain LLM
        self.llm = ChatGroq(
            groq_api_key=None,  # Will use env var
            model_name="llama-3.1-8b-instant"
        )
        
        # Create report directory
        self.report_dir = Path(f"reports_{datetime.now():%Y%m%d_%H%M%S}")
        self.report_dir.mkdir(exist_ok=True)
        
        print(f"Report directory: {self.report_dir}")
    
    def get_user_scenario(self) -> Dict:
        """Get pricing scenario from user interactively"""
        print("\n" + "="*80)
        print("PINN QUANTITATIVE REPORT GENERATOR")
        print("="*80)
        
        print("\nEnter option pricing parameters:")
        
        try:
            S = float(input(f"Spot price S (1-{self.config.S_max}, default 50): ") or "50")
            K = float(input(f"Strike price K (default {self.config.K}): ") or str(self.config.K))
            t = float(input("Current time t in years (0-1, default 0.25): ") or "0.25")
            r = float(input(f"Risk-free rate r (default {self.config.r}): ") or str(self.config.r))
            sigma = float(input(f"Volatility σ (default {self.config.sigma}): ") or str(self.config.sigma))
            
            # Validation
            S = np.clip(S, 1, self.config.S_max)
            t = np.clip(t, 0, self.config.T)
            
            scenario = {
                'S': S,
                'K': K,
                't': t,
                'r': r,
                'sigma': sigma,
                'T': self.config.T
            }
            
            return scenario
        except ValueError:
            print("Invalid input. Using default scenario.")
            return {
                'S': 50.0,
                'K': self.config.K,
                't': 0.25,
                'r': self.config.r,
                'sigma': self.config.sigma,
                'T': self.config.T
            }
    
    def price_option(self, scenario: Dict) -> Dict:
        """Price option and compute Greeks"""
        S = scenario['S']
        t = scenario['t']
        
        # Get PINN prices
        pinn_greeks = self.pricer.greeks(S, t)
        
        # Get analytical prices
        analytical_price = bs_call_price(np.array([S]), scenario['K'], 
                                        scenario['r'], scenario['sigma'], 
                                        scenario['t'], scenario['T'])[0]
        analytical_delta = bs_delta(np.array([S]), scenario['K'], 
                                   scenario['r'], scenario['sigma'], 
                                   scenario['t'], scenario['T'])[0]
        
        results = {
            'scenario': scenario,
            'pinn_price': pinn_greeks['price'],
            'analytical_price': analytical_price,
            'price_error': abs(pinn_greeks['price'] - analytical_price),
            'price_error_pct': abs(pinn_greeks['price'] - analytical_price) / analytical_price * 100,
            'pinn_delta': pinn_greeks['delta'],
            'analytical_delta': analytical_delta,
            'delta_error': abs(pinn_greeks['delta'] - analytical_delta),
            'pinn_gamma': pinn_greeks['gamma']
        }
        
        return results
    
    def generate_price_surface_plot(self) -> str:
        """Generate 3D price surface plot"""
        S_grid = np.linspace(1, self.config.S_max, 100)
        t_grid = np.linspace(0, self.config.T, 50)
        S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)
        
        # PINN prices
        pinn_prices = np.zeros_like(S_mesh)
        for i in range(S_mesh.shape[0]):
            pinn_prices[i, :] = self.pricer.price_batch(S_mesh[i, :], t_mesh[i, 0])
        
        fig = plt.figure(figsize=(14, 5))
        
        # PINN surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(S_mesh, t_mesh, pinn_prices, cmap='viridis')
        ax1.set_xlabel('Asset Price S')
        ax1.set_ylabel('Time t (years)')
        ax1.set_zlabel('Option Price')
        ax1.set_title('PINN Pricing Surface')
        fig.colorbar(surf1, ax=ax1)
        
        # Analytical surface
        analytical_prices = bs_call_price(
            S_mesh.flatten(), self.config.K, self.config.r, 
            self.config.sigma, t_mesh.flatten(), self.config.T
        ).reshape(S_mesh.shape)
        
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(S_mesh, t_mesh, analytical_prices, cmap='plasma')
        ax2.set_xlabel('Asset Price S')
        ax2.set_ylabel('Time t (years)')
        ax2.set_zlabel('Option Price')
        ax2.set_title('Analytical Pricing Surface')
        fig.colorbar(surf2, ax=ax2)
        
        plot_path = self.report_dir / "price_surface.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def generate_comparison_plots(self, results: Dict) -> List[str]:
        """Generate comparison plots"""
        S = results['scenario']['S']
        t = results['scenario']['t']
        
        # Create price comparison
        S_range = np.linspace(1, self.config.S_max, 200)
        pinn_prices = self.pricer.price_batch(S_range, t)
        analytical_prices = bs_call_price(S_range, results['scenario']['K'], 
                                         results['scenario']['r'], 
                                         results['scenario']['sigma'], t, 
                                         results['scenario']['T'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price comparison
        axes[0, 0].plot(S_range, analytical_prices, 'k-', linewidth=2, label='Analytical')
        axes[0, 0].plot(S_range, pinn_prices, 'r--', linewidth=2, label='PINN')
        axes[0, 0].axvline(S, color='blue', linestyle=':', alpha=0.7, label=f'Spot S={S}')
        axes[0, 0].set_xlabel('Asset Price S')
        axes[0, 0].set_ylabel('Call Price')
        axes[0, 0].set_title('Price Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error analysis
        errors = np.abs(pinn_prices - analytical_prices)
        axes[0, 1].plot(S_range, errors, 'g-', linewidth=2)
        axes[0, 1].axvline(S, color='blue', linestyle=':', alpha=0.7)
        axes[0, 1].set_xlabel('Asset Price S')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Pricing Error |PINN - Analytical|')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Delta comparison
        S_range_delta = np.linspace(5, self.config.S_max, 200)
        delta_range = np.array([self.pricer.greeks(s, t)['delta'] for s in S_range_delta])
        analytical_delta = bs_delta(S_range_delta, results['scenario']['K'], 
                                   results['scenario']['r'], 
                                   results['scenario']['sigma'], t, 
                                   results['scenario']['T'])
        
        axes[1, 0].plot(S_range_delta, analytical_delta, 'k-', linewidth=2, label='Analytical')
        axes[1, 0].plot(S_range_delta, delta_range, 'r--', linewidth=2, label='PINN')
        axes[1, 0].axvline(S, color='blue', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('Asset Price S')
        axes[1, 0].set_ylabel('Delta (∂C/∂S)')
        axes[1, 0].set_title('Delta Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text
        summary_text = f"""
PRICING SUMMARY @ S={S:.2f}, t={t:.2f}

PINN Price:           ${results['pinn_price']:.4f}
Analytical Price:     ${results['analytical_price']:.4f}
Absolute Error:       ${results['price_error']:.6f}
Relative Error:       {results['price_error_pct']:.4f}%

PINN Delta:           {results['pinn_delta']:.6f}
Analytical Delta:     {results['analytical_delta']:.6f}
Delta Error:          {results['delta_error']:.6f}

PINN Gamma:           {results['pinn_gamma']:.6f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       family='monospace', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].axis('off')
        
        plot_path = self.report_dir / "comparisons.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return [str(plot_path)]
    
    def generate_llm_analysis(self, results: Dict) -> str:
        """Generate AI-powered analysis using LangChain"""
        # Create analysis prompt
        analysis_prompt = PromptTemplate(
            input_variables=["scenario_data", "pricing_results"],
            template="""
You are a quantitative finance expert. Analyze the following option pricing results 
from a Physics-Informed Neural Network (PINN) model trained on Black-Scholes equation.

SCENARIO:
{scenario_data}

PRICING RESULTS:
{pricing_results}

Provide a professional analysis covering:
1. Model Performance: How well did PINN match analytical results?
2. Greeks Accuracy: Evaluate delta and gamma computation
3. Risk Assessment: What risks should be considered?
4. Trading Implications: What does this pricing tell us?
5. Model Limitations: Any concerns or limitations?

Format as a professional quantitative report section.
"""
        )
        
        # Prepare data for LLM
        scenario_str = json.dumps(results['scenario'], indent=2)
        results_str = f"""
PINN Price: ${results['pinn_price']:.6f}
Analytical Price: ${results['analytical_price']:.6f}
Price Error: {results['price_error_pct']:.4f}%
Delta: {results['pinn_delta']:.6f}
Gamma: {results['pinn_gamma']:.6f}
"""
        
        # Create chain
        chain = analysis_prompt | self.llm | StrOutputParser()
        
        try:
            analysis = chain.invoke({
                "scenario_data": scenario_str,
                "pricing_results": results_str
            })
            return analysis
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return "LLM analysis unavailable. Please check API key."
    
    def generate_full_report(self, results: Dict, llm_analysis: str, 
                            plot_paths: List[str]) -> str:
        """Generate HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PINN Quantitative Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; width: 30%; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #34495e; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PINN Black-Scholes Quantitative Report</h1>
        <p class="timestamp">Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report analyzes option pricing using a Physics-Informed Neural Network (PINN) 
        trained on the Black-Scholes partial differential equation.</p>
        
        <div class="metric">
            <strong>Spot Price (S)</strong><br>${results['scenario']['S']:.2f}
        </div>
        <div class="metric">
            <strong>Strike Price (K)</strong><br>${results['scenario']['K']:.2f}
        </div>
        <div class="metric">
            <strong>Time to Maturity</strong><br>{results['scenario']['T'] - results['scenario']['t']:.2f} years
        </div>
    </div>
    
    <div class="section">
        <h2>Pricing Results</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>PINN</th>
                <th>Analytical</th>
                <th>Error</th>
            </tr>
            <tr>
                <td>Call Price</td>
                <td>${results['pinn_price']:.6f}</td>
                <td>${results['analytical_price']:.6f}</td>
                <td>{results['price_error_pct']:.4f}%</td>
            </tr>
            <tr>
                <td>Delta (∂C/∂S)</td>
                <td>{results['pinn_delta']:.6f}</td>
                <td>{results['analytical_delta']:.6f}</td>
                <td>{abs(results['pinn_delta'] - results['analytical_delta']):.6f}</td>
            </tr>
            <tr>
                <td>Gamma (∂²C/∂S²)</td>
                <td>{results['pinn_gamma']:.6f}</td>
                <td>N/A</td>
                <td>N/A</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
"""
        
        for plot_path in plot_paths:
            html_content += f'<img src="{plot_path}" alt="Analysis Plot">\n'
        
        html_content += f"""
    </div>
    
    <div class="section">
        <h2>AI-Powered Analysis</h2>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
{llm_analysis}
        </pre>
    </div>
    
    <div class="section">
        <h2>Model Parameters</h2>
        <ul>
            <li>Risk-free Rate (r): {results['scenario']['r']:.4f}</li>
            <li>Volatility (σ): {results['scenario']['sigma']:.4f}</li>
            <li>Maturity (T): {results['scenario']['T']:.2f} years</li>
            <li>Neural Network: {len(self.config.layers)} layers</li>
        </ul>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; color: #7f8c8d; border-top: 1px solid #ddd; padding-top: 20px;">
        <p>PINN Quantitative Report Generator | Powered by LangChain & Groq</p>
    </footer>
</body>
</html>
"""
        
        report_path = self.report_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def run(self):
        """Main execution flow"""
        try:
            # Get user scenario
            scenario = self.get_user_scenario()
            print("\n" + "-"*80)
            print(f"Scenario: {json.dumps(scenario, indent=2)}")
            
            # Price option
            print("\nPricing option...")
            results = self.price_option(scenario)
            
            # Generate plots
            print("Generating surface plot...")
            self.generate_price_surface_plot()
            
            print("Generating comparison plots...")
            plot_paths = self.generate_comparison_plots(results)
            
            # Generate LLM analysis
            print("Generating AI analysis (using Groq LLM)...")
            llm_analysis = self.generate_llm_analysis(results)
            
            # Generate final report
            print("Generating HTML report...")
            report_path = self.generate_full_report(results, llm_analysis, plot_paths)
            
            print("\n" + "="*80)
            print("✅ REPORT GENERATED SUCCESSFULLY")
            print("="*80)
            print(f"Report Location: {report_path}")
            print(f"Report Directory: {self.report_dir}")
            print("\nKey Results:")
            print(f"  PINN Price:       ${results['pinn_price']:.6f}")
            print(f"  Analytical Price: ${results['analytical_price']:.6f}")
            print(f"  Pricing Error:    {results['price_error_pct']:.4f}%")
            print(f"  Delta:            {results['pinn_delta']:.6f}")
            print(f"  Gamma:            {results['pinn_gamma']:.6f}")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ==================== MAIN ====================
if __name__ == "__main__":
    generator = QuantReportGenerator()
    generator.run()
