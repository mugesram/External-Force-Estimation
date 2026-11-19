# Common commands. The venv Python runs without ROS's PYTHONPATH (ROS ships NumPy 1.x packages that clash).
PY := env -u PYTHONPATH .venv/bin/python
SEEDS ?= 1-4

.PHONY: help venv test tier0 experiments bench ros-setup tier1-run tier1-ingest tier1-eval clean-runs

help:
	@echo "make venv          create .venv and install the efe package"
	@echo "make test          unit + known-answer tests"
	@echo "make bench         bench identification -> config/priors_bench.yaml, results/bench/"
	@echo "make tier0         Tier 0 (Python sim): all scenarios x filters x seeds -> results/tier0/"
	@echo "make experiments   robustness / ablation / replay Monte Carlo -> results/tier0/experiments/"
	@echo "make ros-setup     px4_msgs + Micro XRCE-DDS Agent + colcon build (one time)"
	@echo "make tier1-run SCEN=S1_main SEED=1   one Isaac + Pegasus + PX4 flight, then ingest"
	@echo "make tier1-eval    evaluate all ingested Tier 1 runs -> results/tier1/"

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -e '.[test]'

test:
	$(PY) -m pytest -q

bench:
	$(PY) -m efe.tools.bench --drag-from runs/tier0/S4_aggressive/seed0

tier0:
	$(PY) -m efe.evaluation.run_matrix --tier 0 --seeds $(SEEDS)

experiments:
	$(PY) -m efe.evaluation.experiments --tier 0 --seeds $(SEEDS)

ros-setup:
	tools/setup_ros.sh

SCEN ?= S1_main
SEED ?= 1
tier1-run:
	tools/run_tier1.sh $(SCEN) $(SEED)

tier1-ingest:
	$(PY) -m efe.io.ingest runs/tier1/*/seed*

tier1-eval:
	$(PY) -m efe.evaluation.run_matrix --tier 1 --seeds $(SEEDS) --scenarios $(SCENARIOS)
