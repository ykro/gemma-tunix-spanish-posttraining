PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
XLA_SIM_FLAGS ?= --xla_force_host_platform_device_count=4

.PHONY: all experiments verify test lint security clean

all: lint test verify

experiments:
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) experiments/01_ephemeral_tpu_mesh.py
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) experiments/02_gemma_nnx_architecture.py
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) experiments/03_lora_surgical_injection.py
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) experiments/04_grpo_alignment_loop.py
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) experiments/05_acid_test_evaluation.py

verify:
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTHON) verify_posttraining.py

test:
	XLA_FLAGS="$(XLA_SIM_FLAGS)" $(PYTEST) tests/

lint:
	$(RUFF) check experiments/ tests/ verify_posttraining.py

security:
	gitleaks detect --source . --no-git

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
