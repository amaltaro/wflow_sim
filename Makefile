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
# Failure rates as percentages: 0, 1, 5, 10, 25
FAILURE_RATES := 0 1 5 10 25

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
	@echo "  simulate-all   - Run simulations for all wallclock times and failure rates"
	@echo "  visualize-all  - Generate visualizations for all wallclock times and failure rates"
	@echo "  all            - Run simulations and visualizations for all combinations"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Use cases configured: $(USE_CASES)"
	@echo "Wallclock times configured: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Failure rates configured: $(FAILURE_RATES)%"
	@echo "Customize by setting variables, e.g.:"
	@echo "  make all USE_CASES='case1_real case2_homo'"
	@echo "  make simulate-all FAILURE_RATES='0 5 10'"

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
# Runs both overhead and nooverhead scenarios automatically for all wallclock times and failure rates
# Results are saved with _overhead.json and _nooverhead.json suffixes
# Results are organized in nested structure: {case_name}/{time_hours}/fr{failure_rate}/
.PHONY: simulate-all
simulate-all:
	@echo "Starting batch simulation for use cases: $(USE_CASES)"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Failure rates: $(FAILURE_RATES)%"
	@echo "Max job slots: $(MAX_JOB_SLOTS)"
	@echo "Running both overhead and nooverhead scenarios"
	@echo ""
	@total_combinations=$$(($$(echo $(USE_CASES) | wc -w) * $$(echo $(WALLCLOCK_TIMES) | wc -w) * $$(echo $(FAILURE_RATES) | wc -w) * 2)); \
	echo "Total simulation combinations: $$total_combinations (use_cases × wallclock_times × failure_rates × 2 overhead modes)"; \
	echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		wallclock_hours=$$(( $$wallclock_time / 3600 )); \
		echo "=========================================="; \
		echo "Simulating with wallclock time: $$wallclock_time seconds ($$wallclock_hours hours)"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			echo "=== Simulating use case: $$use_case ($$wallclock_hours hours) ==="; \
			for failure_rate in $(FAILURE_RATES); do \
				echo "  --- Failure rate: $$failure_rate% ---"; \
				for workflow_file in $(TEMPLATES_DIR)/$$use_case/*_const_*.json; do \
					if [ -f "$$workflow_file" ]; then \
						echo "    Running simulation WITH overhead: $$(basename $$workflow_file) (fr$$failure_rate)"; \
						$(PYTHON) -m src.workflow_runner \
							--target-wallclock-time $$wallclock_time \
							--max-job-slots $(MAX_JOB_SLOTS) \
							--failure-rate $$failure_rate \
							--input-workflow-path $$workflow_file || exit 1; \
						echo "    Running simulation WITHOUT overhead: $$(basename $$workflow_file) (fr$$failure_rate)"; \
						$(PYTHON) -m src.workflow_runner \
							--target-wallclock-time $$wallclock_time \
							--max-job-slots $(MAX_JOB_SLOTS) \
							--failure-rate $$failure_rate \
							--no-overhead \
							--input-workflow-path $$workflow_file || exit 1; \
					fi; \
				done; \
			done; \
			echo "=== Completed use case: $$use_case ($$wallclock_hours hours) ==="; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All simulations completed! (Results saved with _overhead.json and _nooverhead.json suffixes)"
	@echo "Results are organized in nested structure: {case_name}/{time_hours}/fr{failure_rate}/"

# Generate visualizations for all use cases, wallclock times, and failure rates
# Visualization script automatically processes both overhead and nooverhead files
# and generates separate visualizations for each scenario
# Note: Run 'make setup-viz' first if visualization dependencies are not installed
.PHONY: visualize-all
visualize-all:
	@echo "Starting batch visualization for use cases: $(USE_CASES)"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Failure rates: $(FAILURE_RATES)%"
	@echo "Processing both overhead and nooverhead result files"
	@echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		wallclock_hours=$$(( $$wallclock_time / 3600 )); \
		echo "=========================================="; \
		echo "Visualizing results for wallclock time: $$wallclock_time seconds ($$wallclock_hours hours)"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			echo "=== Generating visualizations for use case: $$use_case ($$wallclock_hours hours) ==="; \
			use_case_base_dir="$(RESULTS_DIR)/$$use_case/$${wallclock_hours}h"; \
			if [ -d "$$use_case_base_dir" ]; then \
				for failure_rate in $(FAILURE_RATES); do \
					fr_dir="$$use_case_base_dir/fr$$failure_rate"; \
					if [ -d "$$fr_dir" ]; then \
						output_dir="$(VIZ_OUTPUT_DIR)/$$use_case/$${wallclock_hours}h/fr$$failure_rate"; \
						echo "  Processing results directory: $$fr_dir"; \
						$(PYTHON) scripts/workflow_visualization.py \
							$$fr_dir \
							--output-dir $$output_dir || exit 1; \
					else \
						echo "  Warning: Results directory $$fr_dir not found. Skipping."; \
					fi; \
				done; \
			else \
				echo "  Warning: Results base directory $$use_case_base_dir not found. Skipping."; \
			fi; \
			echo "=== Completed visualizations for use case: $$use_case ($$wallclock_hours hours) ==="; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All visualizations completed!"

# Combined target: run simulations and generate visualizations
# Automatically handles both overhead and nooverhead scenarios for all wallclock times and failure rates
.PHONY: all
all:
	@echo "=========================================="
	@echo "Running complete workflow analysis"
	@echo "Wallclock times: $(WALLCLOCK_TIMES) seconds (1h, 6h, 12h, 18h, 24h)"
	@echo "Use cases: $(USE_CASES)"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/3: Running simulations (both overhead and no-overhead for all wallclock times and failure rates)..."
	@$(MAKE) simulate-all
	@echo ""
	@echo "Step 2/3: Installing visualization dependencies..."
	@$(MAKE) setup-viz
	@echo ""
	@echo "Step 3/3: Generating visualizations (both overhead and no-overhead for all wallclock times and failure rates)..."
	@$(MAKE) visualize-all
	@echo ""
	@echo "=========================================="
	@echo "Complete workflow finished successfully!"
	@echo "Results: $(RESULTS_DIR)/"
	@echo "Visualizations: $(VIZ_OUTPUT_DIR)/"
	@echo "Results are organized in nested structure: {case_name}/{time_hours}/fr{failure_rate}/"
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
