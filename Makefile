# Name of the virtual environment directory
VENV_DIR = .venv
PYTHON = python3.11

HOUR ?= 0
MINUTE ?= 1
SECOND ?= 0

# Default target
all: setup_cpu

# Target for setting up the GPU environment - doesn't seem compatible with my device
# setup_gpu: $(VENV_DIR)/bin/activate requirements-gpu.txt
# 	$(VENV_DIR)/bin/pip install -r requirements-gpu.txt

# Target for setting up the CPU environment
setup_cpu: $(VENV_DIR)/bin/activate requirements-cpu.txt
	$(VENV_DIR)/bin/pip install -r requirements-cpu.txt

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Run Jupyter notebook
jupyter:
	$(VENV_DIR)/bin/jupyter notebook

# Run 1D data
run-1d: setup_cpu
	USE_2D_DATA=False $(VENV_DIR)/bin/python main.py --hour $(HOUR) --minute $(MINUTE) --second $(SECOND)
#	$(MAKE) clean

# Run 2D data
run-2d: setup_cpu
	USE_2D_DATA=True $(VENV_DIR)/bin/python main.py --hour $(HOUR) --minute $(MINUTE) --second $(SECOND)
#	$(MAKE) clean

# Clean up virtual environment
clean:
	rm -rf $(VENV_DIR)

# Phony targets
.PHONY: all setup jupyter clean
