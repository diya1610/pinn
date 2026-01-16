"""
Streamlit Frontend for PINN Quantitative Report Generator
Interactive dashboard for Black-Scholes option pricing using Physics-Informed Neural Networks
"""

import streamlit as st
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'pinn_bs_best.pth'

config = Config()

# ==================== ANALYTICAL SOLUTIONS ====================
def bs_call_price(S: np.ndarray, K: float, r: float, sigma: float, 
                  t: float, T: float) -> np.ndarray:
    """Black-Scholes analytical call option price."""
    tau = T - t
    S = np.array(S, dtype=float)
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    tau = np.maximum(tau, 1e-12)
    
    # Avoid log of zero or negative numbers
    S = np.maximum(S, 1e-10)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    price = S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    price[S <= 1e-10] = 0.0
    
    return price

def bs_delta(S: np.ndarray, K: float, r: float, sigma: float, 
             t: float, T: float) -> np.ndarray:
    """Black-Scholes delta (∂C/∂S)."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    delta = norm.cdf(d1)
    delta[S <= 1e-10] = 0.0
    
    return delta

def bs_gamma(S: np.ndarray, K: float, r: float, sigma: float, 
             t: float, T: float) -> np.ndarray:
    """Black-Scholes gamma (∂²C/∂S²)."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    gamma[S <= 1e-10] = 0.0
    
    return gamma

def bs_vega(S: np.ndarray, K: float, r: float, sigma: float, 
            t: float, T: float) -> np.ndarray:
    """Black-Scholes vega (∂C/∂σ)."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    vega = S * norm.pdf(d1) * np.sqrt(tau)
    
    return vega

def bs_theta(S: np.ndarray, K: float, r: float, sigma: float, 
             t: float, T: float) -> np.ndarray:
    """Black-Scholes theta (∂C/∂t)."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(tau)) - 
             r * K * np.exp(-r*tau) * norm.cdf(d2)) / 365  # Per day
    
    return theta

def bs_rho(S: np.ndarray, K: float, r: float, sigma: float, 
           t: float, T: float) -> np.ndarray:
    """Black-Scholes rho (∂C/∂r)."""
    tau = T - t
    S = np.array(S, dtype=float)
    tau = np.maximum(tau, 1e-12)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    rho = K * tau * np.exp(-r*tau) * norm.cdf(d2)
    
    return rho

# ==================== NORMALIZER ====================
class Normalizer:
    """Input/output normalization for improved training."""
    def __init__(self, S_max: float, T: float):
        self.S_mean = S_max / 2
        self.S_std = S_max / 4
        self.tau_mean = T / 2
        self.tau_std = T / 4
        self.u_mean = S_max / 4
        self.u_std = S_max / 4
    
    def normalize_input(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize [S, tau] inputs."""
        X_norm = X.clone()
        X_norm[:, 0:1] = (X[:, 0:1] - self.S_mean) / self.S_std
        X_norm[:, 1:2] = (X[:, 1:2] - self.tau_mean) / self.tau_std
        return X_norm
    
    def denormalize_output(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize output."""
        return u_norm * self.u_std + self.u_mean

# ==================== PINN ARCHITECTURE ====================
class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.linear(x))

class ImprovedPINN(nn.Module):
    """Enhanced PINN with residual connections."""
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        super().__init__()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()
        
        self.layers_list = nn.ModuleList()
        
        # Input layer
        self.layers_list.append(nn.Linear(layers[0], layers[1]))
        
        # Hidden layers with residual connections
        for i in range(1, len(layers)-2):
            if layers[i] == layers[i+1]:
                self.layers_list.append(ResidualBlock(layers[i], self.activation))
            else:
                self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
        
        # Output layer
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
    """Easy-to-use interface for option pricing after training."""
    
    def __init__(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.normalizer = checkpoint['normalizer']
        
        config_pinn = Config()
        self.model = ImprovedPINN([2, 128, 128, 128, 128, 1], 'tanh')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.config = config_pinn
        self.device = config_pinn.device
        self.model.to(self.device)
    
    def price(self, S: float, t: float) -> float:
        """Price a call option at given spot and time."""
        tau = self.config.T - t
        X = torch.tensor([[S, tau]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_norm = self.normalizer.normalize_input(X)
            u_norm = self.model(X_norm)
            u = self.normalizer.denormalize_output(u_norm)
        
        return u.item()
    
    def greeks(self, S: float, t: float) -> dict:
        """Compute Greeks (Delta, Gamma) at given spot and time."""
        tau = self.config.T - t
        S_tensor = torch.tensor([[S]], dtype=torch.float32, 
                               requires_grad=True).to(self.device)
        tau_tensor = torch.tensor([[tau]], dtype=torch.float32).to(self.device)
        X = torch.cat([S_tensor, tau_tensor], dim=1)
        
        X_norm = self.normalizer.normalize_input(X)
        u_norm = self.model(X_norm)
        u = self.normalizer.denormalize_output(u_norm)
        
        # Delta
        delta = torch.autograd.grad(u, S_tensor, create_graph=True)[0]
        
        # Gamma
        gamma = torch.autograd.grad(delta, S_tensor)[0]
        
        return {
            'price': u.item(),
            'delta': delta.item(),
            'gamma': gamma.item()
        }

# ==================== STREAMLIT CONFIGURATION ====================
st.set_page_config(
    page_title="PINN Quantitative Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #0066cc;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        color: #0066cc;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.pricer = None

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="header">📊 PINN Quantitative Report Generator</div>', 
                unsafe_allow_html=True)
    st.write("Physics-Informed Neural Network for Black-Scholes Option Pricing with LangChain Analysis")
    
    # Sidebar for parameters
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.info(
            "**Model Training Info:**\n\n"
            "PINN was trained with:\n"
            "- K = $50\n"
            "- r = 5%\n"
            "- σ = 25%\n\n"
            "Use these values for best accuracy!"
        )
        
        # Input parameters
        st.markdown("### Option Parameters")
        S = st.slider("Spot Price (S)", 1.0, 150.0, 50.0, step=1.0)
        K = st.slider("Strike Price (K)", 10.0, 150.0, 50.0, step=1.0)
        t = st.slider("Current Time (t) in years", 0.0, 1.0, 0.25, step=0.05)
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, step=0.01)
        sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.25, step=0.05)
        
        st.markdown("### Analysis Type")
        analysis_type = st.radio(
            "Select Analysis",
            ["Quick Pricing", "Greeks Analysis", "Sensitivity Analysis", "Full Report"]
        )
        
        # Load model button
        if st.button("🔄 Load Model", use_container_width=True):
            with st.spinner("Loading PINN model..."):
                try:
                    st.session_state.pricer = PINNPricer(config.checkpoint_path)
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        # Generate report button
        if st.button("📈 Generate Report", use_container_width=True):
            if not st.session_state.model_loaded:
                st.error("Please load the model first!")
            else:
                st.session_state.generate_report = True
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Click 'Load Model' in the sidebar to get started!")
        return
    
    pricer = st.session_state.pricer
    
    # ⚠️ IMPORTANT: Model was trained with K=50, r=5%, σ=25%
    # The model is a function of S and t only, not of K, r, σ
    # Accuracy depends on how close your parameters are to training params
    
    if abs(K - 50.0) > 0.01 or abs(r - 0.05) > 0.001 or abs(sigma - 0.25) > 0.01:
        st.warning(
            f"""
            ⚠️ **Model Parameter Mismatch Detected**
            
            The PINN model was trained with:
            - Strike Price (K): $50.00
            - Risk-Free Rate (r): 5.00%
            - Volatility (σ): 25.00%
            
            But you've set:
            - Strike Price (K): ${K:.2f}
            - Risk-Free Rate (r): {r*100:.2f}%
            - Volatility (σ): {sigma*100:.2f}%
            
            **Accuracy may be reduced for different parameters!**
            
            For best results, use values close to the training parameters.
            """,
            icon="⚠️"
        )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Pricing", "Greeks", "Sensitivity", "Comparison", "Report"]
    )
    
    with tab1:
        st.markdown('<div class="subheader">Option Pricing</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # PINN Price
        with col1:
            pinn_price = pricer.price(S, t)
            st.metric("PINN Price", f"${pinn_price:.4f}", delta=None)
        
        # Analytical Price
        with col2:
            analytical_price = bs_call_price(np.array([S]), K, r, sigma, t, config.T)[0]
            st.metric("Analytical Price", f"${analytical_price:.4f}", delta=None)
        
        # Pricing Error
        with col3:
            if analytical_price > 1e-6:
                error = abs((pinn_price - analytical_price) / analytical_price) * 100
            else:
                error = abs(pinn_price - analytical_price) * 100
            st.metric("Relative Error", f"{error:.2f}%", delta=None)
        
        # Pricing curve
        st.markdown("### Pricing Curve")
        S_range = np.linspace(1, 150, 200)
        tau = config.T - t
        
        pinn_prices = []
        analytical_prices = bs_call_price(S_range, K, r, sigma, t, config.T)
        
        for s in S_range:
            pinn_prices.append(pricer.price(s, t))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(S_range, analytical_prices, 'k-', linewidth=2.5, label='Analytical (Black-Scholes)')
        ax.plot(S_range, pinn_prices, 'r--', linewidth=2.5, label='PINN Model')
        ax.axvline(K, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Strike Price')
        ax.axvline(S, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Current Spot')
        ax.set_xlabel('Asset Price (S)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Option Price', fontsize=12, fontweight='bold')
        ax.set_title(f'Option Pricing at t={t:.2f} years (τ={tau:.2f})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="subheader">Greeks Analysis</div>', unsafe_allow_html=True)
        
        # Compute Greeks
        greeks_pinn = pricer.greeks(S, t)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### PINN Greeks")
            st.metric("Price", f"${greeks_pinn['price']:.4f}")
            st.metric("Delta", f"{greeks_pinn['delta']:.4f}")
            st.metric("Gamma", f"{greeks_pinn['gamma']:.6f}")
        
        with col2:
            st.markdown("### Analytical Greeks")
            analytical_delta = bs_delta(np.array([S]), K, r, sigma, t, config.T)[0]
            analytical_gamma = bs_gamma(np.array([S]), K, r, sigma, t, config.T)[0]
            st.metric("Price", f"${analytical_prices[np.argmin(np.abs(S_range - S))]:.4f}")
            st.metric("Delta", f"{analytical_delta:.4f}")
            st.metric("Gamma", f"{analytical_gamma:.6f}")
        
        with col3:
            st.markdown("### Other Greeks")
            vega = bs_vega(np.array([S]), K, r, sigma, t, config.T)[0]
            theta = bs_theta(np.array([S]), K, r, sigma, t, config.T)[0]
            rho = bs_rho(np.array([S]), K, r, sigma, t, config.T)[0]
            st.metric("Vega", f"{vega:.4f}")
            st.metric("Theta", f"{theta:.4f}")
            st.metric("Rho", f"{rho:.4f}")
        
        # Greeks visualization
        st.markdown("### Greeks Curves")
        S_range_small = np.linspace(max(1, K-50), min(150, K+50), 100)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Delta
        delta_analytical = bs_delta(S_range_small, K, r, sigma, t, config.T)
        delta_pinn = [pricer.greeks(s, t)['delta'] for s in S_range_small]
        axes[0, 0].plot(S_range_small, delta_analytical, 'k-', linewidth=2, label='Analytical')
        axes[0, 0].plot(S_range_small, delta_pinn, 'r--', linewidth=2, label='PINN')
        axes[0, 0].axvline(S, color='blue', linestyle=':', alpha=0.5)
        axes[0, 0].set_title('Delta (∂C/∂S)', fontweight='bold')
        axes[0, 0].set_xlabel('Spot Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gamma
        gamma_analytical = bs_gamma(S_range_small, K, r, sigma, t, config.T)
        gamma_pinn = [pricer.greeks(s, t)['gamma'] for s in S_range_small]
        axes[0, 1].plot(S_range_small, gamma_analytical, 'k-', linewidth=2, label='Analytical')
        axes[0, 1].plot(S_range_small, gamma_pinn, 'r--', linewidth=2, label='PINN')
        axes[0, 1].axvline(S, color='blue', linestyle=':', alpha=0.5)
        axes[0, 1].set_title('Gamma (∂²C/∂S²)', fontweight='bold')
        axes[0, 1].set_xlabel('Spot Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Vega
        vega_range = bs_vega(S_range_small, K, r, sigma, t, config.T)
        axes[0, 2].plot(S_range_small, vega_range, 'b-', linewidth=2.5)
        axes[0, 2].axvline(S, color='blue', linestyle=':', alpha=0.5)
        axes[0, 2].fill_between(S_range_small, vega_range, alpha=0.3)
        axes[0, 2].set_title('Vega (∂C/∂σ)', fontweight='bold')
        axes[0, 2].set_xlabel('Spot Price')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Theta
        theta_range = bs_theta(S_range_small, K, r, sigma, t, config.T)
        axes[1, 0].plot(S_range_small, theta_range, 'g-', linewidth=2.5)
        axes[1, 0].axvline(S, color='blue', linestyle=':', alpha=0.5)
        axes[1, 0].fill_between(S_range_small, theta_range, alpha=0.3, color='green')
        axes[1, 0].set_title('Theta (∂C/∂t)', fontweight='bold')
        axes[1, 0].set_xlabel('Spot Price')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rho
        rho_range = bs_rho(S_range_small, K, r, sigma, t, config.T)
        axes[1, 1].plot(S_range_small, rho_range, 'orange', linewidth=2.5)
        axes[1, 1].axvline(S, color='blue', linestyle=':', alpha=0.5)
        axes[1, 1].fill_between(S_range_small, rho_range, alpha=0.3, color='orange')
        axes[1, 1].set_title('Rho (∂C/∂r)', fontweight='bold')
        axes[1, 1].set_xlabel('Spot Price')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary table
        greeks_data = {
            'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
            'Value': [
                f"{delta_analytical[-1]:.4f}",
                f"{gamma_analytical[-1]:.6f}",
                f"{vega_range[-1]:.4f}",
                f"{theta_range[-1]:.4f}",
                f"{rho_range[-1]:.4f}"
            ]
        }
        axes[1, 2].axis('off')
        table_data = [[greek, value] for greek, value in zip(greeks_data['Greek'], greeks_data['Value'])]
        axes[1, 2].table(cellText=table_data, colLabels=['Greek', 'Value'],
                        cellLoc='center', loc='center', colWidths=[0.5, 0.5])
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="subheader">Sensitivity Analysis</div>', unsafe_allow_html=True)
        
        # Sensitivity to spot price
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Sensitivity to Spot Price")
            S_sensitivity = np.linspace(K-20, K+20, 50)
            prices_pinn = [pricer.price(s, t) for s in S_sensitivity]
            prices_bs = bs_call_price(S_sensitivity, K, r, sigma, t, config.T)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(S_sensitivity, prices_bs, 'k-', linewidth=2.5, label='Analytical')
            ax.plot(S_sensitivity, prices_pinn, 'r--', linewidth=2.5, label='PINN')
            ax.axvline(S, color='blue', linestyle=':', alpha=0.7)
            ax.set_xlabel('Spot Price (S)')
            ax.set_ylabel('Option Price')
            ax.set_title('Price vs Spot Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Price Sensitivity to Volatility")
            sigma_range = np.linspace(0.05, 0.75, 50)
            prices_sigma_pinn = [pricer.price(S, t) for _ in sigma_range]
            prices_sigma_bs = [bs_call_price(np.array([S]), K, r, sig, t, config.T)[0] 
                              for sig in sigma_range]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sigma_range, prices_sigma_bs, 'k-', linewidth=2.5, label='Analytical')
            ax.plot(sigma_range, prices_sigma_pinn, 'r--', linewidth=2.5, label='PINN')
            ax.axvline(sigma, color='blue', linestyle=':', alpha=0.7)
            ax.set_xlabel('Volatility (σ)')
            ax.set_ylabel('Option Price')
            ax.set_title('Price vs Volatility')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="subheader">PINN vs Black-Scholes Comparison</div>', 
                   unsafe_allow_html=True)
        
        # 2D Grid comparison
        S_grid = np.linspace(1, 150, 50)
        t_grid = np.linspace(0, 1, 30)
        S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)
        
        # Calculate prices
        prices_pinn = np.zeros_like(S_mesh)
        prices_bs = np.zeros_like(S_mesh)
        errors = np.zeros_like(S_mesh)
        
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                s, time = S_mesh[i, j], t_mesh[i, j]
                prices_pinn[i, j] = pricer.price(s, time)
                prices_bs[i, j] = bs_call_price(np.array([s]), K, r, sigma, time, config.T)[0]
                errors[i, j] = abs(prices_pinn[i, j] - prices_bs[i, j])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### PINN Pricing Surface")
            fig, ax = plt.subplots(figsize=(10, 8))
            im1 = ax.contourf(S_mesh, t_mesh, prices_pinn, levels=30, cmap='viridis')
            ax.set_xlabel('Spot Price (S)')
            ax.set_ylabel('Time (t)')
            ax.set_title('PINN Prices')
            plt.colorbar(im1, ax=ax, label='Price')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Black-Scholes Surface")
            fig, ax = plt.subplots(figsize=(10, 8))
            im2 = ax.contourf(S_mesh, t_mesh, prices_bs, levels=30, cmap='viridis')
            ax.set_xlabel('Spot Price (S)')
            ax.set_ylabel('Time (t)')
            ax.set_title('Analytical (BS) Prices')
            plt.colorbar(im2, ax=ax, label='Price')
            st.pyplot(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### Absolute Error")
            fig, ax = plt.subplots(figsize=(10, 8))
            im3 = ax.contourf(S_mesh, t_mesh, errors, levels=30, cmap='hot')
            ax.set_xlabel('Spot Price (S)')
            ax.set_ylabel('Time (t)')
            ax.set_title('Absolute Error |PINN - BS|')
            plt.colorbar(im3, ax=ax, label='Error')
            st.pyplot(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<div class="subheader">Quantitative Report</div>', 
                   unsafe_allow_html=True)
        
        # Generate LLM Report
        if st.button("🤖 Generate AI-Powered Report with LangChain"):
            with st.spinner("Generating report using Groq LLM..."):
                try:
                    llm = ChatGroq(
                        model_name="llama-3.1-8b-instant"
                    )
                    
                    # Prepare data for report
                    pinn_price_val = pricer.price(S, t)
                    analytical_price_val = bs_call_price(np.array([S]), K, r, sigma, t, config.T)[0]
                    greeks_val = pricer.greeks(S, t)
                    
                    report_prompt = f"""
                    Imagine u are a quant researcher who works in a company as qualified as citadel Generate a professional quantitative analysis report for the following European Call Option:
                    
                    SCENARIO DETAILS:
                    - Spot Price (S): ${S:.2f}
                    - Strike Price (K): ${K:.2f}
                    - Current Time: {t:.2f} years (T={config.T} years)
                    - Risk-Free Rate (r): {r:.2%}
                    - Volatility (σ): {sigma:.2%}
                    
                    PRICING RESULTS:
                    - PINN Model Price: ${pinn_price_val:.4f}
                    - Analytical (Black-Scholes) Price: ${analytical_price_val:.4f}
                    - Pricing Error: {abs((pinn_price_val - analytical_price_val) / max(analytical_price_val, 1e-6)) * 100:.2f}%
                    
                    GREEKS (Risk Sensitivities):
                    - Delta (Price sensitivity to S): {greeks_val['delta']:.4f}
                    - Gamma (Delta sensitivity to S): {greeks_val['gamma']:.6f}
                    - Vega (Price sensitivity to σ): {bs_vega(np.array([S]), K, r, sigma, t, config.T)[0]:.4f}
                    - Theta (Time decay per day): {bs_theta(np.array([S]), K, r, sigma, t, config.T)[0]:.4f}
                    
                    Please provide:
                    1. Executive Summary of the option position stating if u think its a good buy or not
                    2. Model Performance Analysis (PINN vs Black-Scholes)
                    3. Risk Assessment and Greeks Interpretation and their implications keeping in mind current market conditions which should also be mentioned scrape latest market data from the web
                    4. Trading Recommendations based on Greeks keeping in mind current market conditions which should also be mentioned scrape latest market data from the web
                    5. Conclusions and Risk Warnings
                    
                    Format the report professionally with clear sections.
                    """
                    
                    output_parser = StrOutputParser()
                    chain = llm | output_parser
                    
                    report_text = chain.invoke(report_prompt)
                    
                    st.markdown("### 📋 AI-Generated Report")
                    st.markdown(report_text)
                    
                    # Download report
                    report_filename = f"report_{datetime.now():%Y%m%d_%H%M%S}.txt"
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=report_filename,
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        
        summary_data = {
            'Parameter': [
                'Spot Price (S)',
                'Strike Price (K)',
                'Time to Expiry',
                'Risk-Free Rate',
                'Volatility',
                'PINN Price',
                'Analytical Price',
                'Delta',
                'Gamma'
            ],
            'Value': [
                f"${S:.2f}",
                f"${K:.2f}",
                f"{config.T - t:.2f} years",
                f"{r:.2%}",
                f"{sigma:.2%}",
                f"${pricer.price(S, t):.4f}",
                f"${bs_call_price(np.array([S]), K, r, sigma, t, config.T)[0]:.4f}",
                f"{greeks_pinn['delta']:.4f}",
                f"{greeks_pinn['gamma']:.6f}"
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

if __name__ == "__main__":
    main()
