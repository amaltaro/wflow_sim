# Dependencies

Core simulation code can use mostly the standard library, but the repository also
includes analysis scripts and tests that need a small set of third-party packages.
See `requirements.txt` for exact pins.

## Runtime (batch / examples)

- **`networkx`** – graph utilities where used
- **`numpy`**, **`pandas`**, **`matplotlib`** – analysis scripts under `scripts/`
  (e.g. failure rate and workflow-type sensitivity plots) and any test that imports
  those modules

## Development and testing

- **`pytest>=7.0.0`**

## Installation

```bash
pip install -r requirements.txt
```

### Tests

```bash
pytest tests/ -v
```

**CI** (`.github/workflows/test.yml`) installs `requirements.txt` before `pytest`, so
everything listed there must cover test imports (including `failure_rate_analysis` →
`matplotlib`).

## Python Version

Use **Python 3.10+** in practice; CI uses **3.12** (see the workflow file).
