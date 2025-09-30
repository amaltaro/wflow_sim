# Dependencies

This project has minimal dependencies and uses only Python standard library modules.

## Runtime Dependencies

**None required** - The project uses only Python standard library modules:
- `json` - JSON file handling
- `logging` - Logging functionality  
- `typing` - Type hints
- `dataclasses` - Data classes
- `pathlib` - Path handling
- `time` - Time functions

## Optional Dependencies

For development and testing:
- `pytest>=7.0.0` - Testing framework

## Installation

### Basic Usage
No installation required - just run the scripts directly:
```bash
python examples/metrics_example.py
```

### With Testing
If you want to run the tests:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Python Version

Requires Python 3.8 or higher (for dataclasses and typing features).
