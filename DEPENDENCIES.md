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

### Recommended (uv + Makefile)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+:

```bash
make setup
```

This creates `.venv/` and runs `uv pip install -r requirements.txt`.

### Alternative (pip, matches CI)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Tests

```bash
make test
# or: uv run pytest tests/ -v
```

**CI** (`.github/workflows/test.yml`) installs `requirements.txt` before `pytest`, so
everything listed there must cover test imports (including `failure_rate_analysis` →
`matplotlib`).

## Python Version

Use **Python 3.10+** in practice; CI uses **3.12** (see the workflow file).

Analysis and visualization scripts under `scripts/` use the same packages as tests
(`matplotlib`, `numpy`, `pandas`) — all covered by `requirements.txt`. Run `make setup`
once before batch targets such as `make visualize-all` or `make all`.
