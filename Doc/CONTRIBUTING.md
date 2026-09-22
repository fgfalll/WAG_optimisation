# Contribution Guidelines

## Testing

Run all tests with coverage:
```bash
python -m pytest --cov=co2eor_optimizer --cov-report=html tests/
```

View coverage report:
```bash
open htmlcov/index.html
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Document public APIs with docstrings
- Keep line length under 100 characters

## Pull Requests

1. Create feature branch
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit PR with description of changes

## Development Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-cov  # Ensure test dependencies are installed
```

3. Verify installation:
```bash
python -c "import pytest; print(f'pytest version: {pytest.__version__}')"