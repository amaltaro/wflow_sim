# Results Organization Strategy

## Directory Structure

Results are organized in a nested structure that supports multi-dimensional analysis:

```
results/sim/others/
  case1_real/
    12h/
      fr0/          # failure_rate = 0%
        case1_real_const_001_overhead.json
        case1_real_const_001_nooverhead.json
        case1_real_const_002_overhead.json
        ...
      fr1/          # failure_rate = 1%
        case1_real_const_001_overhead.json
        ...
      fr5/          # failure_rate = 5%
      fr10/         # failure_rate = 10%
      fr25/         # failure_rate = 25%
    6h/
      fr0/
      fr1/
      ...
    24h/
      fr0/
      ...
  case2_homo/
    12h/
      fr0/
      ...
  case3_hetero/
    ...
```

## Structure Benefits

1. **Easy Single-Dimension Iteration**
   - All failure rates for a given case/time: `case1_real/12h/fr*/`
   - All times for a given case/failure rate: `case1_real/*/fr0/`
   - All cases for a given time/failure rate: `*/12h/fr0/`

2. **Clear Hierarchy**
   - Case → Time → Failure Rate → Files
   - Easy to understand and navigate

3. **Compatible with Existing Code**
   - Recursive glob (`**/*.json`) still works
   - Visualization scripts can process entire directories

4. **Scalable**
   - Easy to add more dimensions (e.g., `case1_real/12h/fr0/slots100/`)

5. **Preserves Intermediate Directories**
   - Maintains structure like `others/` if present in template path
   - Supports multiple categories of workflows

## Implementation

The new structure is automatically created by the updated `_get_output_path()` function in:
- `src/workflow_simulator.py`
- `src/workflow_runner.py`

Both functions now accept a `failure_rate` parameter and organize results accordingly.

## Analysis Patterns

### Pattern 1: Compare Failure Rates (Fixed Case & Time)
```bash
# Visualize case1_real with 12h target, all failure rates
# This will process all fr* subdirectories
python scripts/workflow_visualization.py results/sim/others/case1_real/12h
```

### Pattern 2: Compare Target Times (Fixed Case & Failure Rate)
```bash
# Visualize case1_real with fr0, all target times
# Process the entire case directory - visualization script will find all time directories
python scripts/workflow_visualization.py results/sim/others/case1_real
# Then filter or group by time in visualization script
```

### Pattern 3: Compare Cases (Fixed Time & Failure Rate)
```bash
# Visualize all cases with 12h/fr0
# Process parent directory containing all cases
python scripts/workflow_visualization.py results/sim/others
# Then filter or group by case in visualization script
```

### Pattern 4: Single Parameter Combination
```bash
# Visualize specific combination: case1_real, 12h, fr5
python scripts/workflow_visualization.py results/sim/others/case1_real/12h/fr5
```

## Running Simulations with Failure Rate

```bash
# Run simulation with specific failure rate
python -m src.workflow_runner \
  --input-workflow-path templates/others/case1_real/case1_real_const_001.json \
  --target-wallclock-time 43200 \
  --failure-rate 5

# Results will be saved to:
# results/sim/others/case1_real/12h/fr5/case1_real_const_001_overhead.json
```

## Migration from Old Structure

**Old structure**: `results/sim/others/case1_real_12h/case1_real_const_001_overhead.json`

**New structure**: `results/sim/others/case1_real/12h/fr0/case1_real_const_001_overhead.json`

### Migration Options

1. **Manual Migration**: Move files to new structure manually
2. **Script Migration**: Create a migration script to reorganize existing results
3. **Fresh Start**: Re-run simulations with new structure (recommended for consistency)

### Example Migration Script (if needed)

```python
# migrate_results.py
from pathlib import Path
import shutil
import re

def migrate_old_structure():
    """Migrate from case1_real_12h/ to case1_real/12h/fr0/ structure."""
    old_base = Path("results/sim/others")
    
    for old_dir in old_base.glob("*_*h"):  # e.g., case1_real_12h
        # Extract case name and time
        match = re.match(r"(.+)_(\d+)h", old_dir.name)
        if match:
            case_name, hours = match.groups()
            new_base = old_base / case_name / f"{hours}h" / "fr0"
            new_base.mkdir(parents=True, exist_ok=True)
            
            # Move all JSON files
            for json_file in old_dir.glob("*.json"):
                shutil.move(str(json_file), str(new_base / json_file.name))
            
            # Remove old directory if empty
            try:
                old_dir.rmdir()
            except OSError:
                pass
```

## Recommendations

1. **For New Simulations**: Use the new structure automatically (already implemented)
2. **For Analysis**: Update visualization scripts to handle the nested structure
3. **For Batch Processing**: Update Makefile to include failure rate loops
4. **For Documentation**: Keep this file updated as structure evolves
