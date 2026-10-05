PY ?= .venv/bin/python
SEEDS ?= 42 43 44
MODE ?= online
DIR ?= results/online

ONLINE_FAILURE_STRATEGIES = STATIC_MAX STATIC_MEAN STATIC_MEDIAN WEIGHTED_STATIC_MEAN \
	DYNAMIC_HEUR_EQUAL_START DYNAMIC_HEUR_CALIBRATED_START \
	DYNAMIC_META_EQUAL_START DYNAMIC_META_CALIBRATED_START
OFFLINE_STRATEGIES = BASELINE_AutoEncoder BASELINE_LSTMAD BASELINE_LOF BASELINE_IForest \
	STATIC_MAX STATIC_MEAN STATIC_MEDIAN WEIGHTED_STATIC_MEAN

.PHONY: install tune-online tune-offline run-online run-offline run-seeds \
	verify analyze-online analyze-offline replot-weights stats clean

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -e .

tune-online:
	$(PY) scripts/tune_hyperparams.py --mode online

tune-offline:
	$(PY) scripts/tune_hyperparams.py --mode offline

run-online:
	$(PY) scripts/run_experiment.py --config-name online

run-offline:
	$(PY) scripts/run_experiment.py --config-name offline

run-seeds:
	for seed in $(SEEDS); do \
		$(PY) scripts/run_experiment.py --config-name $(MODE) seed=$$seed \
			output_dir=results/$(MODE)_seed_$$seed || exit 1; \
	done

verify:
	$(PY) scripts/verify_results.py $(DIR)

analyze-online:
	$(PY) scripts/analyze.py results/online --case clean
	$(PY) scripts/analyze.py results/online --case failures --strategies $(ONLINE_FAILURE_STRATEGIES)

analyze-offline:
	$(PY) scripts/analyze.py results/offline --case clean --strategies $(OFFLINE_STRATEGIES)

replot-weights:
	$(PY) scripts/replot_weights.py $(DIR)/dynamic_weights --output-dir $(DIR)

stats:
	$(PY) scripts/stats.py results/$(MODE)_seed_*/results.csv

clean:
	rm -rf outputs multirun
	find src scripts -name __pycache__ -type d -prune -exec rm -rf {} +
