#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = virtual_mouse
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 virtual_mouse
	isort --check --diff --profile black virtual_mouse
	black --check --config pyproject.toml virtual_mouse

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml virtual_mouse

## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://my_virtual_mouse/data/ \
		data/

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://my_virtual_mouse/data

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) virtual_mouse/dataset.py

## Run Virtual Mouse using hand gestures
.PHONY: run_virtual_mouse
run_virtual_mouse:
	$(PYTHON_INTERPRETER) virtual_mouse/gesture_control.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
