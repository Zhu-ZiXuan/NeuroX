# Copyright (c) 2024 ZHU ZiXuan
# SPDX-License-Identifier: MIT


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

RUFF_TARGET_DIRS := neurox validations tests

.PHONY: format
format: ## Run `ruff` formatting with auto fix
	uv run ruff format $(RUFF_TARGET_DIRS)
	uv run ruff check --fix-only $(RUFF_TARGET_DIRS)

.PHONY: lint
lint: ## Run `ruff` linting
	uv run ruff check $(RUFF_TARGET_DIRS)

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
