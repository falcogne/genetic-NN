# Name of the virtual environment directory
VENV_DIR = .venv
PYTHON = python3.11

# Default target
all: setup_cpu

# Target for setting up the GPU environment
setup_gpu: $(VENV_DIR)/bin/activate requirements-gpu.txt
	$(VENV_DIR)/bin/pip install -r requirements-gpu.txt

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
	USE_2D_DATA=False $(VENV_DIR)/bin/python main.py
	# $(MAKE) clean

# Run 2D data
run-2d: setup_gpu
	USE_2D_DATA=True $(VENV_DIR)/bin/python main.py
	# $(MAKE) clean

# Clean up virtual environment
clean:
	rm -rf $(VENV_DIR)

# Phony targets
.PHONY: all setup jupyter clean
