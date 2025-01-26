# Name of the virtual environment directory
VENV_DIR = .venv
PYTHON = python3

# Default target
all: setup

# Create virtual environment and install dependencies
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	touch $(VENV_DIR)/bin/activate

# Run Jupyter notebook
jupyter:
	$(VENV_DIR)/bin/jupyter notebook

# Clean up virtual environment
clean:
	rm -rf $(VENV_DIR)

# Phony targets
.PHONY: all setup jupyter clean
