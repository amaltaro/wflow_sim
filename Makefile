# Makefile for DAGFlowSim Project

# FIXME: This ensures we use the correct Python environment
PYTHON := /Users/amaltar2/pyenv-3.12/bin/python
PIP := /Users/amaltar2/pyenv-3.12/bin/pip
PYTEST := /Users/amaltar2/pyenv-3.12/bin/pytest

# Simulation configuration
# Target job lengths: 15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h (seconds)
WALLCLOCK_TIMES := 900 1800 3600 7200 14400 28800 43200 86400
TARGET_WALLCLOCK_TIME := 43200  # Default for single run (12h)
MAX_JOB_SLOTS := -1
# Failure rates as percentages: 0, 1, 5, 10, 25
FAILURE_RATES := 0 1 5 10 25

# Data transfer rate dimension (MB/s and directory names; MBps/GBps = bytes per second)
DATA_TRANSFER_RATE_MBPS := 10 100 1000 10000
DATA_TRANSFER_RATE_DIRS := 10MBps 100MBps 1GBps 10GBps
DATA_TRANSFER_RATE_FR := 0

# Workflow use cases to simulate (modify this list as needed)
USE_CASES := case1_real case2_homo case3_hetero

# Template base directory
TEMPLATES_DIR := templates/others

# Simulation results output directory (results saved as .json; overhead always applied)
RESULTS_DIR := results/sim/others
# Visualization output directory
VIZ_OUTPUT_DIR := results/vis/others
# Construction metrics analysis: fixed scenario (12h, fr5, 100MBps, all workflow types)
CONSTRUCTION_METRICS_TIME := 12h
CONSTRUCTION_METRICS_FR := fr5
CONSTRUCTION_METRICS_FR_LIST := fr0 fr5 fr25
CONSTRUCTION_METRICS_RATE := 100MBps
CONSTRUCTION_METRICS_OUTPUT := results/analysis/construction_metrics

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup          - Set up the project environment"
	@echo "  test           - Run tests"
	@echo "  build-workflows - Build all workflow constructions from generic templates"
	@echo "  run            - Run single workflow simulation"
	@echo "  simulate-all   - Run simulations (all times, failure rates, data rates)"
	@echo "  visualize-all  - Generate visualizations (all times, failure rates, data rates)"
	@echo "  all            - Run simulations and visualizations for all combinations"
	@echo ""
	@echo "Analysis targets:"
	@echo "  analyze-failure-rate           - Analyze failure rate impact across all workflow types"
	@echo "  analyze-workflow-type-sensitivity - Analyze workflow type sensitivity (12h, fr0/fr5/fr25)"
	@echo "  analyze-target-job-length      - Analyze target job length optimization (all workflow types, fr0/fr5/fr25)"
	@echo "  analyze-data-transfer-rate    - Analyze data transfer rate sensitivity (12h, fr0 & fr5)"
	@echo "  analyze-construction-metrics  - Multi-metric construction comparison (12h, fr0/fr5/fr25, 100MBps, all types)"
	@echo ""
	@echo "Cleanup targets:"
	@echo "  clean          - Clean up all generated files"
	@echo "  clean-viz      - Clean only visualization outputs"
	@echo "  clean-results  - Clean only simulation results"
	@echo ""
	@echo "Use cases configured: $(USE_CASES)"
	@echo "Target job lengths: ${WALLCLOCK_TIMES} seconds"
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

# Build all workflow constructions from generic templates
# Input: templates/generic/<case>.json
# Output: templates/others/<case>/<case>_const_*.json and compositions_summary.json
.PHONY: build-workflows
build-workflows:
	@echo "Building workflow constructions from generic templates..."
	@for use_case in $(USE_CASES); do \
		echo "" && echo "=== Building $$use_case ==="; \
		$(PYTHON) -m src.workflow_builder --input templates/generic/$$use_case.json \
			--output templates/others/$$use_case; \
	done
	@echo "Done building all workflow constructions."

# Run single workflow simulation
.PHONY: run
run:
	@echo "Running workflow simulation..."
	$(PYTHON) -m src.workflow_runner --target-wallclock-time $(TARGET_WALLCLOCK_TIME) --input-workflow-path templates/others/case1_real/case1_real_const_001.json

# Run simulations for all configured use cases
# All simulations include overhead (taskset bootstrap and remote I/O).
# Output: results/sim/others/<case>/<time>/fr<fr>/<data_rate>/
.PHONY: simulate-all
simulate-all:
	@echo "Starting batch simulation for use cases: $(USE_CASES)"
	@echo "Target job lengths: ${WALLCLOCK_TIMES} seconds"
	@echo "Failure rates: $(FAILURE_RATES)%"
	@echo "Data transfer rates: $(DATA_TRANSFER_RATE_DIRS)"
	@echo "Max job slots: $(MAX_JOB_SLOTS)"
	@echo ""
	@total_combinations=$$(($$(echo $(USE_CASES) | wc -w) * $$(echo $(WALLCLOCK_TIMES) | wc -w) * $$(echo $(FAILURE_RATES) | wc -w) * 4)); \
	echo "Total simulation combinations: $$total_combinations"; \
	echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		time_dir=$$(awk -v t=$$wallclock_time 'BEGIN{if(t<3600) printf "%dm", t/60; else printf "%dh", t/3600}'); \
		echo "=========================================="; \
		echo "Simulating with target job length: $$wallclock_time seconds ($$time_dir)"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			echo "=== Simulating use case: $$use_case ($$time_dir) ==="; \
			for failure_rate in $(FAILURE_RATES); do \
				for rate_dir in $(DATA_TRANSFER_RATE_DIRS); do \
					case $$rate_dir in \
						10MBps) rate=10;; \
						100MBps) rate=100;; \
						1GBps) rate=1000;; \
						10GBps) rate=10000;; \
						*) rate=100;; \
					esac; \
					echo "  --- Failure rate: $$failure_rate%, data rate: $$rate_dir ---"; \
					for workflow_file in $(TEMPLATES_DIR)/$$use_case/*_const_*.json; do \
						if [ -f "$$workflow_file" ]; then \
							echo "    Running: $$(basename $$workflow_file) (fr$$failure_rate, $$rate_dir)"; \
							$(PYTHON) -m src.workflow_runner \
								--target-wallclock-time $$wallclock_time \
								--max-job-slots $(MAX_JOB_SLOTS) \
								--failure-rate $$failure_rate \
								--data-transfer-rate $$rate \
								--input-workflow-path $$workflow_file || exit 1; \
						fi; \
					done; \
				done; \
			done; \
			echo "=== Completed use case: $$use_case ($$time_dir) ==="; \
		echo ""; \
		done; \
		echo ""; \
	done
	@echo "All simulations completed!"

# Generate visualizations for all use cases, target job lengths, failure rates, and data rates.
# Output: results/vis/others/<case>/<time>/fr<fr>/<data_rate>/
.PHONY: visualize-all
visualize-all:
	@echo "Starting batch visualization for use cases: $(USE_CASES)"
	@echo "Target job lengths: ${WALLCLOCK_TIMES} seconds"
	@echo "Failure rates: $(FAILURE_RATES)%"
	@echo "Data transfer rates: $(DATA_TRANSFER_RATE_DIRS)"
	@echo ""
	@for use_case in $(USE_CASES); do \
		echo "=========================================="; \
		echo "=== Generating visualizations for use case: $$use_case ==="; \
		echo "=========================================="; \
		for wallclock_time in $(WALLCLOCK_TIMES); do \
			time_dir=$$(awk -v t=$$wallclock_time 'BEGIN{if(t<3600) printf "%dm", t/60; else printf "%dh", t/3600}'); \
			echo "=== Visualizing results for target job length: $$wallclock_time seconds ($$time_dir) ==="; \
			use_case_base_dir="$(RESULTS_DIR)/$$use_case/$$time_dir"; \
			if [ -d "$$use_case_base_dir" ]; then \
				for failure_rate in $(FAILURE_RATES); do \
					for rate_dir in $(DATA_TRANSFER_RATE_DIRS); do \
						fr_dir="$$use_case_base_dir/fr$$failure_rate/$$rate_dir"; \
						if [ -d "$$fr_dir" ]; then \
							output_dir="$(VIZ_OUTPUT_DIR)/$$use_case/$$time_dir/fr$$failure_rate/$$rate_dir"; \
							echo "" && echo "====> Processing: $$fr_dir"; \
							$(PYTHON) scripts/workflow_visualization.py $$fr_dir --output-dir $$output_dir || exit 1; \
						fi; \
						if [ ! -d "$$fr_dir" ]; then \
							echo "" && echo "====> WARNING: Results directory $$fr_dir not found. Skipping."; \
						fi; \
					done; \
				done; \
			else \
				echo "" && echo "====> WARNING: Results base directory $$use_case_base_dir not found. Skipping."; \
			fi; \
			echo "=== Completed visualizations for use case: $$use_case ($$time_dir) ==="; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All visualizations completed!"

# Generate failure rate impact analysis (cross-dimensional comparison)
# Analyzes how all 16 constructions perform across different failure rates
.PHONY: analyze-failure-rate
analyze-failure-rate:
	@echo "Starting failure rate impact analysis"
	@echo "Use cases: $(USE_CASES)"
	@echo "Target job lengths: ${WALLCLOCK_TIMES} seconds"
	@echo ""
	@for wallclock_time in $(WALLCLOCK_TIMES); do \
		time_dir=$$(awk -v t=$$wallclock_time 'BEGIN{if(t<3600) printf "%dm", t/60; else printf "%dh", t/3600}'); \
		echo "=========================================="; \
		echo "Analyzing failure rate impact for target job length: $$time_dir"; \
		echo "=========================================="; \
		for use_case in $(USE_CASES); do \
			echo "" && echo "*** Analyzing use case: $$use_case ($$time_dir) ***"; \
			use_case_base_dir="$(RESULTS_DIR)/$$use_case/$$time_dir"; \
			if [ -d "$$use_case_base_dir" ]; then \
				$(PYTHON) scripts/failure_rate_analysis.py \
					$(RESULTS_DIR) \
					$$use_case \
					$$time_dir || exit 1; \
			else \
				echo "  Warning: Results base directory $$use_case_base_dir not found. Skipping."; \
			fi; \
			echo "*** Completed analysis for use case: $$use_case ($$time_dir) ***"; \
			echo ""; \
		done; \
		echo ""; \
	done
	@echo "All failure rate impact analyses completed!"

# Generate workflow type sensitivity analysis (cross-dimensional comparison)
# Analyzes how different workflow types respond to hybrid constructions
# Runs for failure rates: 0%, 5%, 25% (fr0, fr5, fr25)
.PHONY: analyze-workflow-type-sensitivity
analyze-workflow-type-sensitivity:
	@echo "Starting workflow type sensitivity analysis"
	@echo "Use cases: $(USE_CASES)"
	@echo "Configuration: 12h target, failure rates fr0 (0%), fr5 (5%), fr25 (25%)"
	@echo ""
	@for failure_rate in fr0 fr5 fr25; do \
		echo "*** Failure rate: $$failure_rate ***"; \
		$(PYTHON) scripts/workflow_type_sensitivity.py \
			$(RESULTS_DIR) \
			12h \
			$$failure_rate || exit 1; \
		echo ""; \
	done
	@echo "Workflow type sensitivity analysis completed!"

# Generate target job length optimization analysis (cross-dimensional comparison)
# Analyzes how different workflow constructions perform across target job lengths
.PHONY: analyze-target-job-length
analyze-target-job-length:
	@echo "Starting target job length optimization analysis"
	@echo "Use cases: $(USE_CASES)"
	@echo "Failure rates: fr0 (0%), fr5 (5%), fr25 (25%)"
	@echo ""
	@for use_case in $(USE_CASES); do \
		echo "*** Analyzing use case: $$use_case ***"; \
		use_case_base_dir="$(RESULTS_DIR)/$$use_case"; \
		if [ -d "$$use_case_base_dir" ]; then \
			for failure_rate in fr0 fr5 fr25; do \
				echo "  --- Failure rate: $$failure_rate ---"; \
				$(PYTHON) scripts/target_job_length_analysis.py \
					$(RESULTS_DIR) \
					$$use_case \
					$$failure_rate || exit 1; \
			done; \
		else \
			echo "  Warning: Results base directory $$use_case_base_dir not found. Skipping."; \
		fi; \
		echo "*** Completed analysis for use case: $$use_case ***"; \
		echo ""; \
	done
	@echo "All target job length optimization analyses completed!"

# Analyze data transfer rate sensitivity (run after simulate-all or simulate-data-transfer-rate).
# Reads from unified tree $(RESULTS_DIR)/.../12h/<fr>/<data_rate>/, writes to results/analysis/data_transfer_rate/<fr>/
# Runs for failure rates: fr0 (0%), fr5 (5%)
.PHONY: analyze-data-transfer-rate
analyze-data-transfer-rate:
	@echo "Starting data transfer rate sensitivity analysis"
	@echo "Input: $(RESULTS_DIR) (12h, fr0 & fr5, all data rates)"
	@echo "Output: results/analysis/data_transfer_rate/<failure_rate>/"
	@echo ""
	@for failure_rate in fr0 fr5; do \
		echo "*** Failure rate: $$failure_rate ***"; \
		$(PYTHON) scripts/data_transfer_rate_analysis.py \
			$(RESULTS_DIR) \
			--failure-rate $$failure_rate \
			--output-dir results/analysis/data_transfer_rate/$$failure_rate || exit 1; \
		echo ""; \
	done
	@echo "Data transfer rate analysis completed!"

# Construction metrics analysis: 12h, fr0/fr5/fr25, 100MBps, all 3 workflow types
# Policies: default (throughput), io_prioritized, resource_prioritized
# Output: results/analysis/construction_metrics/<use_case>/12h/<fr>/100MBps/
.PHONY: analyze-construction-metrics
analyze-construction-metrics:
	@echo "Starting construction metrics analysis"
	@echo "Scenario: job length=$(CONSTRUCTION_METRICS_TIME), failure rates=$(CONSTRUCTION_METRICS_FR_LIST), data rate=$(CONSTRUCTION_METRICS_RATE)"
	@echo "Use cases: $(USE_CASES)"
	@echo "Policies: default, io_prioritized, resource_prioritized"
	@echo ""
	@for failure_rate in $(CONSTRUCTION_METRICS_FR_LIST); do \
		echo "*** Failure rate: $$failure_rate ***"; \
		for use_case in $(USE_CASES); do \
			sim_dir="$(RESULTS_DIR)/$$use_case/$(CONSTRUCTION_METRICS_TIME)/$$failure_rate/$(CONSTRUCTION_METRICS_RATE)"; \
			output_dir="$(CONSTRUCTION_METRICS_OUTPUT)/$$use_case/$(CONSTRUCTION_METRICS_TIME)/$$failure_rate/$(CONSTRUCTION_METRICS_RATE)"; \
			if [ -d "$$sim_dir" ]; then \
				echo "  Processing: $$use_case"; \
				for policy in default io_prioritized resource_prioritized; do \
					echo "    Policy: $$policy"; \
					$(PYTHON) scripts/construction_metrics_analysis.py \
						$$sim_dir \
						--output-dir $$output_dir \
						--scenario-label "$$use_case $(CONSTRUCTION_METRICS_TIME) $$failure_rate $(CONSTRUCTION_METRICS_RATE)" \
						--policy $$policy || exit 1; \
				done; \
			else \
				echo "  Warning: $$sim_dir not found. Skipping $$use_case."; \
			fi; \
		done; \
		echo ""; \
	done
	@echo "Construction metrics analysis completed!"
	@echo "Output: $(CONSTRUCTION_METRICS_OUTPUT)/"

# Combined target: run simulations and generate visualizations
.PHONY: all
all:
	@echo "=========================================="
	@echo "Running complete workflow analysis"
	@echo "Target job lengths: ${WALLCLOCK_TIMES} seconds"
	@echo "Use cases: $(USE_CASES)"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/3: Running simulations (overhead always applied)..."
	@$(MAKE) simulate-all
	@echo ""
	@echo "Step 2/3: Installing visualization dependencies..."
	@$(MAKE) setup-viz
	@echo ""
	@echo "Step 3/3: Generating visualizations..."
	@$(MAKE) visualize-all
	@echo ""
	@echo "=========================================="
	@echo "Complete workflow finished successfully!"
	@echo "Results: $(RESULTS_DIR)/"
	@echo "Visualizations: $(VIZ_OUTPUT_DIR)/"
	@echo "Results are organized in nested structure: {case_name}/{time_dir}/fr{failure_rate}/"
	@echo "=========================================="

# Clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	@echo "Removing simulation result files..."
	find results/sim -name "*.json" -type f -delete 2>/dev/null || true
	find results -name "*_overhead.json" -type f -delete 2>/dev/null || true
	rm -rf $(VIZ_OUTPUT_DIR)/
	rm -rf results/analysis/data_transfer_rate/
	rm -rf $(CONSTRUCTION_METRICS_OUTPUT)/
	@echo "Cleanup complete!"

# Clean only visualization outputs
.PHONY: clean-viz
clean-viz:
	@echo "Cleaning visualization outputs..."
	rm -rf $(VIZ_OUTPUT_DIR)/
	@echo "Visualization cleanup complete!"

# Clean only simulation results
.PHONY: clean-results
clean-results:
	@echo "Cleaning simulation results..."
	find results/sim -name "*.json" -type f -delete 2>/dev/null || true
	find results -name "*_overhead.json" -type f -delete 2>/dev/null || true
	@echo "Results cleanup complete!"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r requirements.txt
