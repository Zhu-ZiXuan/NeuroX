# Copyright (c) 2024 ZHU ZiXuan
# SPDX-License-Identifier: MIT


PYTHON ?= python


# --- Help Info ---

.DEFAULT_GOAL := help
.PHONY: help
help: ## Show this help message
	@echo "Usage: make <target>"
	@awk 'BEGIN {FS = ": .*## "; i = 0; max_len = 0; file = ""} \
		/^[a-zA-Z0-9_-]+:.*?## / { \
			targets[i] = $$1; help_msgs[i] = $$2; files[i] = FILENAME; len = length($$1); \
			if (len > max_len) max_len = len; \
			i++; \
		} END { \
			for (j = 0; j < i; j++) { \
				if (files[j] != file) { file = files[j]; printf "\n\033[33m%s:\033[0m\n", file; } \
				printf "  \033[36m%-" max_len "s\033[0m %s\n", targets[j], help_msgs[j]; \
			} \
		}' $(MAKEFILE_LIST)


# --- Develop Tools ---

RUFF_TARGET_DIRS := neurox example validations tests

.PHONY: format
format: ## Run `ruff` formatting with auto fix
	uv run ruff format $(RUFF_TARGET_DIRS)
	uv run ruff check --fix-only $(RUFF_TARGET_DIRS)

.PHONY: lint
lint: ## Run `ruff` linting
	uv run ruff check $(RUFF_TARGET_DIRS) | tee ruff_report.log

MYPY_TARGET_DIRS := neurox

.PHONY: check
check: ## Run `mypy`
	uv run mypy $(MYPY_TARGET_DIRS) | tee mypy_report.log

PYTEST_TARGET_DIRS ?= tests

.PHONY: test
test: ## Run `pytest`
	uv run pytest $(PYTEST_TARGET_DIRS)

.PHONY: docs
docs: ## Run `mkdocs`
	uv run mkdocs build --strict

.PHONY: clean
clean: ## Clean up cache and temporary files
	rm -rf .*_cache *.log site .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +


# --- Device Requirement ---

.PHONY: require-device
require-device:
	@test -n "$(DEVICE)" || { echo "DEVICE is required, for example DEVICE=cuda:0 or DEVICE=cpu" >&2; exit 2; }


# --- Validation campaigns ---

.PHONY: validate_xue2020jssc
validate_xue2020jssc: require-device ## Run the xue2020jssc SINWP 1T1R CIM sub-array validation campaign
	uv run python -m validations.xue2020jssc.validate --device $(DEVICE)

.PHONY: validate_ye2023jssc
validate_ye2023jssc: require-device ## Run the ye2023jssc WH-2T1R CIM macro validation campaign
	uv run python -m validations.ye2023jssc.validate --device $(DEVICE)


# --- Examples ---

MAX_SAMPLES ?= 1

train-lenet: DATASET_DIR ?= dataset/mnist
train-lenet: RAW_CKPT ?= weight/lenet_float.pth
train-lenet: BATCH_SIZE ?= 128
train-lenet: EPOCHS ?= 50
train-lenet: LR ?= 1e-2

.PHONY: train-lenet
train-lenet: require-device ## Float-train LeNet-5 on MNIST
	$(PYTHON) -m example.lenet.train_float --dataset-dir $(DATASET_DIR) --checkpoint $(RAW_CKPT) \
		--device $(DEVICE) --batch-size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LR)

train-bert: DATASET_DIR ?= dataset/sst2
train-bert: RAW_CKPT ?= weight/bert_small_float.pth
train-bert: BATCH_SIZE ?= 32
train-bert: MAX_LENGTH ?= 128
train-bert: EPOCHS ?= 3
train-bert: LR ?= 2e-5

.PHONY: train-bert
train-bert: require-device ## Float-fine-tune BERT-small on SST-2
	$(PYTHON) -m example.bert.train_float --dataset-dir $(DATASET_DIR) --checkpoint $(RAW_CKPT) \
		--device $(DEVICE) --batch-size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LR) \
		--max-length $(MAX_LENGTH)

eval-lenet: EVAL_CKPT ?= weight/lenet_float.pth
eval-lenet: PRESET ?= xue2020jssc
.PHONY: eval-lenet
eval-lenet: require-device ## Observe LeNet energy while retaining the original numerical output
	$(PYTHON) -m example.lenet.evaluate --checkpoint $(EVAL_CKPT) --preset $(PRESET) \
		--device $(DEVICE) --num-samples $(if $(MAX_SAMPLES),$(MAX_SAMPLES),1) --output log/energy/lenet_$(PRESET).json

eval-bert: EVAL_CKPT ?= weight/bert_small_float.pth
eval-bert: PRESET ?= xue2020jssc
eval-bert: MAX_LENGTH ?= 128
.PHONY: eval-bert
eval-bert: require-device ## Observe BERT energy while retaining the original numerical output
	$(PYTHON) -m example.bert.evaluate --checkpoint $(EVAL_CKPT) --preset $(PRESET) \
		--device $(DEVICE) --num-samples $(if $(MAX_SAMPLES),$(MAX_SAMPLES),1) \
		--sequence-length $(MAX_LENGTH) --output log/energy/bert_$(PRESET).json
