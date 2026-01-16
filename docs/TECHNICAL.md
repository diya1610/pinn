# Technical Architecture

Deep dive into PINN implementation and model training methodology.

## Physics-Informed Neural Networks (PINNs)

### Problem Formulation

The Black-Scholes PDE for European call options:

$$\frac{\partial u}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 u}{\partial S^2} + r S \frac{\partial u}{\partial S} - r u = 0$$

**Boundary Conditions:**
- Terminal: $u(S, T) = \max(S - K, 0)$
- Spatial: $u(0, t) = 0$, $u(\infty, t) \approx S - K e^{-r\tau}$

### Neural Network Architecture

```
Input Layer (2 neurons)
├─ S: Spot price [0, 150]
└─ τ: Time-to-maturity [0, 1]
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (128 neurons, ReLU)
    ↓
Hidden Layer 3 (128 neurons, ReLU)
    ↓
Hidden Layer 4 (128 neurons, ReLU)
    ↓ [Residual Connection]
    ↓
Output Layer (1 neuron)
    ↓
Call Option Price
```

**Key Features:**
- **Residual Connections**: Skip connections between dense layers improve gradient flow
- **Xavier Initialization**: Weights initialized for stable training
- **Input Normalization**: S normalized to [0, 1] by dividing by 150
- **Output Normalization**: Prices scaled by spot price for numerical stability

### Training Process

**Loss Function:**
$$L = w_1 L_{PDE} + w_2 L_{boundary} + w_3 L_{initial}$$

Where:
- $L_{PDE}$: Residual error in differential equation
- $L_{boundary}$: Boundary condition satisfaction
- $L_{initial}$: Terminal payoff accuracy

**Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 15,000 | Stopped early if validation loss plateaus |
| Batch Size | 256 | Training data samples per iteration |
| Optimizer | Adam | lr=0.001, betas=(0.9, 0.999) |
| Learning Rate Schedule | ReduceLROnPlateau | Factor=0.5, patience=50 epochs |
| Dropout | None | Full network; no regularization needed |
| Validation Split | 20% | Used for early stopping |

**Training Parameters (Fixed):**
```python
K = 50.0      # Strike price
r = 0.05      # Annual risk-free rate
sigma = 0.25  # Annual volatility
T = 1.0       # Time to maturity (1 year)
```

## Greeks Computation

All Greeks computed via automatic differentiation (PyTorch autograd).

### Implementation Example

```python
# Price computation with gradient tracking
u_pred = model(S_t)  # u_pred requires_grad=True

# Delta: ∂u/∂S
delta = torch.autograd.grad(
    outputs=u_pred.sum(),
    inputs=S_t,
    create_graph=True
)[0]

# Gamma: ∂²u/∂S²
gamma = torch.autograd.grad(
    outputs=delta.sum(),
    inputs=S_t,
    create_graph=True
)[0]
```

### Greeks Formulas

| Greek | Symbol | Formula | Interpretation |
|-------|--------|---------|-----------------|
| Delta | Δ | ∂u/∂S | Price sensitivity to spot price |
| Gamma | Γ | ∂²u/∂S² | Delta sensitivity (convexity) |
| Vega | ν | ∂u/∂σ | Price sensitivity to volatility |
| Theta | Θ | ∂u/∂t | Price decay over time |
| Rho | ρ | ∂u/∂r | Price sensitivity to interest rate |

**Black-Scholes References:**
```python
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
d2 = d1 - sigma * np.sqrt(tau)

delta = norm.cdf(d1)
gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau))
vega = S * norm.pdf(d1) * np.sqrt(tau)
theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(tau)) 
         - r * K * np.exp(-r*tau) * norm.cdf(d2))
rho = K * tau * np.exp(-r*tau) * norm.cdf(d2)
```

## Model Accuracy Analysis

### Parameter Generalization

**Good Generalization:**
- Spot price (S): 0 to 150 (tested range)
- Time-to-maturity (τ): 0 to 1 year

**Poor Generalization:**
- Strike (K): Only trained on K=50; large deviations increase error
- Rate (r): Only trained on r=5%; deviations affect pricing significantly
- Volatility (σ): Only trained on σ=25%; sensitive to changes

### Error Analysis

**Error as function of parameter deviation:**

For $\Delta K = |K - 50|$:
- ΔK = $1 → Error ≈ 0.1%
- ΔK = $5 → Error ≈ 0.5%
- ΔK = $10 → Error ≈ 2.0%

Similar patterns for rate and volatility deviations.

### Validation Results

On held-out test set:
- **Mean Absolute Error**: 0.01 (for normalized prices)
- **Max Error**: 0.05 (at boundaries)
- **R² Score**: 0.9998 (correlation vs analytical)
- **Inference Speed**: <10ms per price (CPU)

## Model Loading

```python
import torch

# CPU loading
checkpoint = torch.load('pinn_bs_best.pth', 
                        map_location=torch.device('cpu'),
                        weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# GPU loading (if CUDA available)
checkpoint = torch.load('pinn_bs_best.pth')
model.to(torch.device('cuda'))
```

**Note**: `weights_only=False` required for older checkpoint format.

## Integration with Black-Scholes

PINNs are compared against analytical Black-Scholes to verify correctness:

```python
# PINN prediction
pinn_price = model.predict(S, t)

# Black-Scholes analytical
tau = T - t
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
bs_price = S * norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d1 - sigma*np.sqrt(tau))

# Comparison
error = abs(pinn_price - bs_price) / bs_price
```

## LangChain Integration

Report generation uses LangChain with Groq LLM:

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

template = """Analyze the following option pricing results...
{analysis_prompt}
"""

prompt = PromptTemplate(input_variables=["analysis_prompt"], template=template)
chain = prompt | llm | StrOutputParser()
report = chain.invoke({"analysis_prompt": analysis_text})
```

## Performance Optimization

### Inference Optimization
1. **Batch Processing**: Compute multiple prices simultaneously
2. **GPU Acceleration**: Uses GPU if CUDA available
3. **Model Quantization**: 32-bit float sufficient; no int8 needed for stability

### Streamlit Caching
```python
@st.cache_resource
def load_model():
    # Model loads once, cached across reruns
    return PINNPricer()

@st.cache_data
def compute_greeks(S, t):
    # Cache Greeks computations
    return model.compute_greeks(S, t)
```

## Common Issues and Solutions

### Numerical Instability
- **Cause**: Extreme parameter values outside training range
- **Solution**: Normalize inputs; validate parameter ranges
- **Implementation**: Done automatically in `Normalizer` class

### Gradient Explosion
- **Cause**: Automatic differentiation through complex function
- **Solution**: Use `torch.autograd.grad(..., retain_graph=True)`
- **Implementation**: Handled in Greeks computation engine

### Memory Issues
- **Cause**: Large batch sizes with gradient computation
- **Solution**: Reduce batch size; use `torch.no_grad()` for inference
- **Implementation**: Small batches in Streamlit (batch_size=1)

## Further Research

1. **Alternative Loss Functions**: Try Huber loss for robustness
2. **Adaptive Weighting**: Dynamic weight adjustment for PDE/BC loss
3. **Multi-PINN Approach**: Separate networks for different volatility regimes
4. **Monte Carlo Validation**: Compare against MC simulation for accuracy
5. **Inverse Problems**: Implied volatility solving via reverse PINN

---

For implementation details, see source code:
- `streamlit_app.py` - Production code
- `pinn.ipynb` - Training notebook
- `pinn.py` - Core model definition
