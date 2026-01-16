# Contributing to PINN Option Pricing

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/pinn-option-pricing.git
cd pinn-option-pricing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest
```

## Code Standards

- **Style**: Follow PEP 8 (use `black` for formatting)
- **Linting**: Run `flake8` before submitting
- **Docstrings**: Use Google-style docstrings for all functions
- **Type hints**: Include type hints for function parameters and returns

```python
def compute_greek(option_price: np.ndarray, param: float) -> np.ndarray:
    """Compute Greek by numerical differentiation.
    
    Args:
        option_price: PINN model output
        param: Parameter to differentiate with respect to
        
    Returns:
        Greek value array
    """
    pass
```

## Contribution Types

### 🐛 Bug Reports
1. Check existing issues first
2. Include minimal reproducible example
3. Specify Python version, OS, and dependency versions
4. Include error traceback

### ✨ Feature Requests
1. Open an issue describing the feature
2. Explain use case and expected behavior
3. Consider impact on project scope

### 📝 Documentation
- Improve README or doc files
- Add docstrings to undocumented code
- Create tutorials or examples

### 🔧 Code Improvements
- Optimize inference speed
- Improve numerical stability
- Add support for new option types (American, Asian, etc.)
- Enhance visualization

## Pull Request Process

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make changes** with meaningful commits
3. **Run tests**: `pytest` to ensure no regressions
4. **Format code**: `black streamlit_app.py quant_report_generator.py`
5. **Lint**: `flake8 streamlit_app.py quant_report_generator.py`
6. **Push**: `git push origin feature/your-feature-name`
7. **Open PR**: Include description of changes and reference related issues

## Commit Message Guidelines

```
[TYPE] Brief description (50 chars max)

Detailed explanation if needed (wrap at 72 chars)

- Bullet points for multiple changes
- Reference issue numbers: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Areas for Contribution

| Area | Priority | Notes |
|------|----------|-------|
| American options support | High | Requires modification to loss function |
| GPU optimization | Medium | Profile current bottlenecks first |
| Implied volatility solver | Medium | Reverse problem solving |
| API endpoint | Low | FastAPI wrapper around Streamlit |
| Docker containerization | Low | Production deployment helper |

## Questions?

- Check existing GitHub issues
- Review documentation in `docs/` folder
- Look at main application files for code patterns

Thanks for contributing! 🚀
