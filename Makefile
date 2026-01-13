# Makefile for DAGFlowSim Project

# FIXME: This ensures we use the correct Python environment
PYTHON := /Users/amaltar2/pyenv-3.12/bin/python
PIP := /Users/amaltar2/pyenv-3.12/bin/pip
PYTEST := /Users/amaltar2/pyenv-3.12/bin/pytest

# Simulation configuration
# Wallclock times in seconds: 1h, 6h, 12h, 18h, 24h
WALLCLOCK_TIMES := 3600 21600 43200 64800 86400
TARGET_WALLCLOCK_TIME := 43200  # Default for single run
MAX_JOB_SLOTS := -1

# Workflow use cases to simulate (modify this list as needed)
USE_CASES := case1_real case2_homo case3_hetero

# Template base directory
TEMPLATES_DIR := templates/others

# Simulation results output directory
# Note: Results are saved with _overhead.json or _nooverhead.json suffixes
RESULTS_DIR := results/sim/others
# Visualization output directory
VIZ_OUTPUT_DIR := results/vis/others

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup          - Set up the project environment"
	@echo "  test           - Run tests"
	@echo "  run            - Run single workflow simulation"
	@echo "  simulate-all   - Run simulations for all wallclock times (1h, 6h, 12h, 18h, 24h)"
	@echo "  visualize-all  - Generate visualizations for all wallclock times"
	@echo "  all            - Run simulations and visualizations for all wallclock times"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Use cases configured: $(USE_CASES)"
	@echo "Wallclock times configured: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Customize by setting USE_CASES variable, e.g.:"
	@echo "  make all USE_CASES='case1_real case2_homo'"

# Set up the project environment
.PHONY: setup
setup:
	@echo "Setting up project environment..."
	$(PYTHON) -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "Environment setup complete!"

# Install visualization dependencies
.PHONY: setup-viz
setup-viz:
	@echo "Installing visualization dependencies..."
	$(PIP) install -r requirements_visualization.txt
	@echo "Visualization dependencies installed!"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v

# Run single workflow simulation
.PHONY: run
run:
	@echo "Running workflow simulation..."
	$(PYTHON) -m src.workflow_runner --target-wallclock-time $(TARGET_WALLCLOCK_TIME) --input-workflow-path templates/others/case1_real/case1_real_const_001.json

# Run simulations for all configured use cases
# Runs both overhead and nooverhead scenarios automatically for all wallclock times
# Results are saved with _overhead.json and _nooverhead.json suffixes
# Directory names include wallclock time suffix (e.g., case1_real_6h, case1_real_12h)
.PHONY: simulate-all
simulate-all:
	@echo "Starting batch simulation for use cases: $(USE_CASES)"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Max job slots: $(MAX_JOB_SLOTS)"
	@echo "Running both overhead and nooverhead scenarios"
	@echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		wallclock_hours=$$(( $$wallclock_time / 3600 )); \
		echo "=========================================="; \
		echo "Simulating with wallclock time: $$wallclock_time seconds ($$wallclock_hours hours)"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			echo "=== Simulating use case: $$use_case ($$wallclock_hours hours) ==="; \
			for workflow_file in $(TEMPLATES_DIR)/$$use_case/*_const_*.json; do \
				if [ -f "$$workflow_file" ]; then \
					echo "  Running simulation WITH overhead: $$workflow_file"; \
					$(PYTHON) -m src.workflow_runner \
						--target-wallclock-time $$wallclock_time \
						--max-job-slots $(MAX_JOB_SLOTS) \
						--input-workflow-path $$workflow_file || exit 1; \
					echo "  Running simulation WITHOUT overhead: $$workflow_file"; \
					$(PYTHON) -m src.workflow_runner \
						--target-wallclock-time $$wallclock_time \
						--max-job-slots $(MAX_JOB_SLOTS) \
						--no-overhead \
						--input-workflow-path $$workflow_file || exit 1; \
				fi; \
			done; \
			echo "=== Completed use case: $$use_case ($$wallclock_hours hours) ==="; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All simulations completed! (Results saved with _overhead.json and _nooverhead.json suffixes)"
	@echo "Results are organized in directories with wallclock time suffixes (e.g., case1_real_6h, case1_real_12h)"

# Generate visualizations for all use cases and wallclock times
# Visualization script automatically processes both overhead and nooverhead files
# and generates separate visualizations for each scenario
# Note: Run 'make setup-viz' first if visualization dependencies are not installed
.PHONY: visualize-all
visualize-all:
	@echo "Starting batch visualization for use cases: $(USE_CASES)"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Processing both overhead and nooverhead result files"
	@echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		wallclock_hours=$$(( $$wallclock_time / 3600 )); \
		echo "=========================================="; \
		echo "Visualizing results for wallclock time: $$wallclock_time seconds ($$wallclock_hours hours)"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			use_case_dir="$${use_case}_$${wallclock_hours}h"; \
			echo "=== Generating visualizations for use case: $$use_case ($$wallclock_hours hours) ==="; \
			if [ -d "$(RESULTS_DIR)/$$use_case_dir" ]; then \
				echo "  Processing results directory: $(RESULTS_DIR)/$$use_case_dir"; \
				$(PYTHON) scripts/workflow_visualization.py \
					$(RESULTS_DIR)/$$use_case_dir \
					--output-dir $(VIZ_OUTPUT_DIR)/$$use_case_dir || exit 1; \
			else \
				echo "  Warning: Results directory $(RESULTS_DIR)/$$use_case_dir not found. Skipping."; \
			fi; \
			echo "=== Completed visualizations for use case: $$use_case ($$wallclock_hours hours) ==="; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All visualizations completed!"

# Combined target: run simulations and generate visualizations
# Automatically handles both overhead and nooverhead scenarios for all wallclock times
.PHONY: all
all:
	@echo "=========================================="
	@echo "Running complete workflow analysis"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Use cases: $(USE_CASES)"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/3: Running simulations (both overhead and no-overhead for all wallclock times)..."
	@$(MAKE) simulate-all
	@echo ""
	@echo "Step 2/3: Installing visualization dependencies..."
	@$(MAKE) setup-viz
	@echo ""
	@echo "Step 3/3: Generating visualizations (both overhead and no-overhead for all wallclock times)..."
	@$(MAKE) visualize-all
	@echo ""
	@echo "=========================================="
	@echo "Complete workflow finished successfully!"
	@echo "Results: $(RESULTS_DIR)/"
	@echo "Visualizations: $(VIZ_OUTPUT_DIR)/"
	@echo "Results are organized by wallclock time (e.g., case1_real_6h, case1_real_12h)"
	@echo "=========================================="

# Clean up generated files
# Removes both _overhead.json and _nooverhead.json files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	@echo "Removing all result files (*_overhead.json and *_nooverhead.json)..."
	find results -name "*_overhead.json" -type f -delete 2>/dev/null || true
	find results -name "*_nooverhead.json" -type f -delete 2>/dev/null || true
	rm -rf $(VIZ_OUTPUT_DIR)/
	@echo "Cleanup complete!"

# Clean only visualization outputs
.PHONY: clean-viz
clean-viz:
	@echo "Cleaning visualization outputs..."
	rm -rf $(VIZ_OUTPUT_DIR)/
	@echo "Visualization cleanup complete!"

# Clean only simulation results
# Removes both _overhead.json and _nooverhead.json files
.PHONY: clean-results
clean-results:
	@echo "Cleaning simulation results..."
	@echo "Removing all result files (*_overhead.json and *_nooverhead.json)..."
	find results -name "*_overhead.json" -type f -delete 2>/dev/null || true
	find results -name "*_nooverhead.json" -type f -delete 2>/dev/null || true
	@echo "Results cleanup complete!"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt
