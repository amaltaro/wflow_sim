# Results Organization Strategy

## Directory Structure

Results are organized in a nested structure that supports multi-dimensional analysis:

```
results/sim/others/
  seq_real/
    12h/
      fr0/                    # failure_rate = 0%
        10MBps/               # data transfer rate = 10 MB/s (bytes per second)
          seq_real_const_001.json
          seq_real_const_002.json
          ...
        100MBps/              # 100 MB/s
        1GBps/                 # 1 GB/s
        10GBps/                # 10 GB/s
      fr1/                    # failure_rate = 1%
        10MBps/
        100MBps/
        ...
      fr5/
      fr10/
      fr25/
    6h/
      fr0/
        10MBps/
        ...
    24h/
      fr0/
      ...
  seq_homo/
    12h/
      fr0/
        ...
  seq_hetero/
    ...
```

The same structure is used for visualizations under `results/vis/others/`:
`results/vis/others/<workflow_type>/<target_job_length>/<failure_rate>/<data_rate>/`.

Each simulation JSON file (e.g. `seq_real_const_001.json`) contains three top-level keys: **`metrics`** (workflow-level metrics, including aggregated job totals such as `total_job_overhead_secs`), **`simulation_result`** (groups, job sample, `failure_rate` intended, `actual_failure_rate` observed, `total_job_retries`, etc.), and **`simulation_stats`** (distribution statistics over all jobs: mean, std, median, min, max, n for job overhead, events per job `batch_size`, and I/O metrics total_write_local_mb, total_write_remote_mb, total_read_local_mb, total_read_remote_mb, for use in visualizations and error bars).

## Structure Benefits

1. **Easy Single-Dimension Iteration**
   - All failure rates for a given case/time/data rate: `seq_real/12h/fr*/100MBps/`
   - All data rates for a given case/time/failure rate: `seq_real/12h/fr0/*/`
   - All times for a given case/failure rate/data rate: `seq_real/*/fr0/100MBps/`
   - All cases for a given time/failure rate/data rate: `*/12h/fr0/100MBps/`

2. **Clear Hierarchy**
   - Case → Time → Failure Rate → Data Rate → Files
   - Data rate dirs (10MBps, 100MBps, 1GBps, 10GBps) use uppercase B for bytes per second
   - Easy to understand and navigate

3. **Compatible with Existing Code**
   - Recursive glob (`**/*.json`) still works
   - Visualization scripts can process entire directories

4. **Scalable**
   - Data transfer rate is already a dimension (`seq_real/12h/fr0/100MBps/`)
   - Further dimensions can be added if needed

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
# Visualize seq_real with 12h target, all failure rates
# This will process all fr* subdirectories
python scripts/workflow_visualization.py results/sim/others/seq_real/12h
```

### Pattern 2: Compare Target Times (Fixed Case & Failure Rate)
```bash
# Visualize seq_real with fr0, all target times
# Process the entire case directory - visualization script will find all time directories
python scripts/workflow_visualization.py results/sim/others/seq_real
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
# Visualize specific combination: seq_real, 12h, fr5
python scripts/workflow_visualization.py results/sim/others/seq_real/12h/fr5
```

## Running Simulations with Failure Rate and Data Transfer Rate

```bash
# Run simulation with specific failure rate and data transfer rate (default 100 MB/s)
python -m src.workflow_runner \
  --input-workflow-path templates/others/seq_real/seq_real_const_001.json \
  --target-wallclock-time 43200 \
  --failure-rate 5 \
  --data-transfer-rate 100

# Results will be saved to:
# results/sim/others/seq_real/12h/fr5/100MBps/seq_real_const_001.json
```

## Migration from Old Structure

**Old structure**: `results/sim/others/seq_real_12h/seq_real_const_001.json`

**New structure**: `results/sim/others/seq_real/12h/fr0/100MBps/seq_real_const_001.json`

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
    """Migrate from seq_real_12h/ to seq_real/12h/fr0/ structure."""
    old_base = Path("results/sim/others")
    
    for old_dir in old_base.glob("*_*h"):  # e.g., seq_real_12h
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
