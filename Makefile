# Copyright (c) 2024 ZHU ZiXuan
# SPDX-License-Identifier: MIT


PYTHON ?= python


.DEFAULT_GOAL := help


# --- develop ---

RUFF_TARGET_DIR := neurox example tests

.PHONY: format
format: ## Run `ruff` formatter with auto fix
	uv run ruff format $(RUFF_TARGET_DIR)
	uv run ruff check --fix-only $(RUFF_TARGET_DIR)

.PHONY: lint
lint: ## Run `ruff` linter
	uv run ruff check $(RUFF_TARGET_DIR) 2>&1 | tee ruff_report.log

MYPY_TARGET_DIR := neurox

.PHONY: check
check: ## Run `mypy` static analysis
	uv run mypy $(MYPY_TARGET_DIR) 2>&1 | tee mypy_report.log

PYTEST_DIRS ?= tests

.PHONY: test
test: ## Run pytest
	uv run pytest $(PYTEST_DIRS)


# --- Examples ---
#
# Two reference pipelines: LeNet-5 on MNIST and BERT-small on SST-2.
# All per-target defaults (dataset, checkpoints, hyperparameters) are
# declared via GNU Make target-specific variables.  A small, shared
# vocabulary (DATASET_DIR, RAW_CKPT, HAT_CKPT, EPOCHS, LR, ...) is
# reused across both models; each target binds its own default, and
# command-line overrides still win, e.g.:
#
#   make hat-lenet EPOCHS=30 LR=3e-5
#   make eval-bert MAX_SAMPLES=1000
#
# Hardware knobs: HAT uses the bundled 1T1R TOML and the lossless
# ideal xbar; evaluate defaults to ``physical`` for deployment
# accuracy.  Override ``CIM_MACRO=...`` to swap.

DEVICE      ?= cuda:0
MAX_SAMPLES ?=

# LeNet-5 on MNIST
train-lenet hat-lenet eval-lenet: DATASET_DIR ?= dataset/mnist
train-lenet hat-lenet eval-lenet: RAW_CKPT    ?= weight/lenet_float.pth
train-lenet hat-lenet eval-lenet: HAT_CKPT    ?= weight/lenet_hat.pth
train-lenet hat-lenet eval-lenet: BATCH_SIZE  ?= 128
train-lenet:                      EPOCHS      ?= 50
train-lenet:                      LR          ?= 1e-2
hat-lenet:                        EPOCHS      ?= 80
hat-lenet:                        LR          ?= 2e-5
hat-lenet:                        CAL_BATCHES ?= 128
hat-lenet:                        KD_ALPHA    ?= 0.1
hat-lenet:                        KD_TEMP     ?= 2.0
hat-lenet:                        CIM_MACRO   ?= ideal
eval-lenet:                       CIM_MACRO   ?= physical
eval-lenet:                       EVAL_CKPT   ?= weight/lenet_qat.pth
eval-lenet:                       CONFIG      ?= macro_with_physical_xbar.toml
eval-lenet:                       POLICY      ?= macro_with_physical_xbar.policy.toml

.PHONY: train-lenet
train-lenet: ## Float-train LeNet-5 on MNIST
	$(PYTHON) -m example.lenet.train --dataset-dir $(DATASET_DIR) --checkpoint $(RAW_CKPT) \
		--device $(DEVICE) --batch-size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LR)

.PHONY: hat-lenet
hat-lenet: ## Hardware-aware QAT for LeNet-5 (macro-in-the-loop, KD)
	$(PYTHON) -m example.lenet.hat_qat --dataset-dir $(DATASET_DIR) \
		--float-checkpoint $(RAW_CKPT) --checkpoint $(HAT_CKPT) \
		--device $(DEVICE) --cim_macro $(CIM_MACRO) \
		--batch-size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LR) \
		--calibration-batches $(CAL_BATCHES) \
		--kd-alpha $(KD_ALPHA) --kd-temperature $(KD_TEMP)

.PHONY: eval-lenet
eval-lenet: ## Evaluate a LeNet QAT checkpoint on MNIST
	$(PYTHON) -m example.lenet.evaluate --dataset-dir $(DATASET_DIR) \
		--checkpoint $(EVAL_CKPT) --config $(CONFIG) --policy $(POLICY) \
		--device $(DEVICE) --cim_macro $(CIM_MACRO) --batch-size $(BATCH_SIZE) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))

# BERT-small on SST-2
hat-bert eval-bert: DATASET_DIR ?= dataset/sst2
hat-bert eval-bert: RAW_CKPT    ?= weight/bert_small_float.pth
hat-bert eval-bert: HAT_CKPT    ?= weight/bert_small_hat.pth
hat-bert eval-bert: BATCH_SIZE  ?= 16
hat-bert eval-bert: MAX_LENGTH  ?= 128
hat-bert:           EPOCHS      ?= 10
hat-bert:           LR          ?= 1e-4
hat-bert:           CAL_BATCHES ?= 128
hat-bert:           KD_ALPHA    ?= 0.5
hat-bert:           KD_TEMP     ?= 2.0
hat-bert:           KD_HIDDEN   ?= 0.3
hat-bert:           CIM_MACRO        ?= ideal
eval-bert:          CIM_MACRO        ?= physical
eval-bert:          EVAL_CKPT   ?= weight/bert_small_qat.pth
eval-bert:          CONFIG      ?= macro.toml
eval-bert:          POLICY      ?= macro.policy.toml

.PHONY: hat-bert
hat-bert: ## Hardware-aware QAT for BERT-small on SST-2 (KD + pooled MSE)
	$(PYTHON) -m example.bert.hat_qat --dataset-dir $(DATASET_DIR) \
		--float-checkpoint $(RAW_CKPT) --checkpoint $(HAT_CKPT) \
		--device $(DEVICE) --cim_macro $(CIM_MACRO) \
		--batch-size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LR) \
		--max-length $(MAX_LENGTH) --calibration-batches $(CAL_BATCHES) \
		--kd-alpha $(KD_ALPHA) --kd-temperature $(KD_TEMP) --kd-hidden-weight $(KD_HIDDEN)

.PHONY: eval-bert
eval-bert: ## Evaluate a BERT-small QAT checkpoint on SST-2
	$(PYTHON) -m example.bert.evaluate --dataset-dir $(DATASET_DIR) \
		--checkpoint $(EVAL_CKPT) --config $(CONFIG) --policy $(POLICY) \
		--device $(DEVICE) --cim_macro $(CIM_MACRO) \
		--batch-size $(BATCH_SIZE) --max-length $(MAX_LENGTH) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))


# --- Validation campaigns ---

.PHONY: validate_xue2020jssc
validate_xue2020jssc: ## Run the xue2020jssc SINWP 1T1R CIM sub-array validation campaign
	TORCH_COMPILE_DISABLE=1 uv run python validations/xue2020jssc/validate.py

.PHONY: validate_ye2023jssc
validate_ye2023jssc: ## Run the ye2023jssc WH-2T1R CIM macro validation campaign
	TORCH_COMPILE_DISABLE=1 uv run python validations/ye2023jssc/validate.py


# --- Documentation ---

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	mkdocs serve

.PHONY: docs-build
docs-build: ## Build documentation site
	mkdocs build --strict


# --- Cleaning ---

.PHONY: clean
clean: ## Clean up cache and temporary files
	rm -rf .*_cache *.log site .coverage htmlcov
# 	rm -rf ~/.triton/cache /tmp/torchinductor_*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +


# --- Help Info ---

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
