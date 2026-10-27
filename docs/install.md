# Installation Guide

Welcome to the installation guide for `torch_structure`. To ensure a smooth installation process, please follow the steps below.

---

## Prerequisites

`torch_structure` requires **PyTorch**.

!!! warning Important Note on Dependencies
    PyTorch requires a specific build depending on your system (e.g., CUDA version if you have a GPU). As a result, it cannot be automatically installed as part of the package dependencies and must be installed manually. All other dependencies, including PyTorch Geometric, are installed automatically.

---

## Installation Steps

### 1. (Optional, Recommended) Create a Virtual Environment
To avoid conflicts with other Python packages and to maintain a clean working environment, we highly recommend creating a new Python virtual environment.

For instance, with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```bash
conda create -n torch_structure_env python=3.12
conda activate torch_structure_env
conda install pip
```

or, with [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.12  
```

### 2. Install PyTorch

Visit the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to find the correct command for your system. We recommend installing the latest stable version.

For example, if using CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install `torch_structure`

Install the `torch_structure` package via pip:

```bash
pip install torch_structure
```

!!! info "Alternative: Install directly from GitHub"
    To install the very latest version of `torch_structure`, clone the [GitHub repository](https://github.com/lazlo-bleker/torch-structure):

    ```bash
    git clone https://github.com/lazlo-bleker/torch-structure.git
    ```

    Followed by creating an editable installation using the following command from within the cloned repository:

    ```bash
    pip install -e .  
    ```

!!! tip "Optional: PyTorch Geometric extensions"
    `torch_structure` does not require the compiled PyTorch Geometric extensions (`torch_scatter`, `pyg_lib`, ...). If they are installed, PyTorch Geometric uses them automatically where they are faster. See the [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for how to install them.

### 4. Verify the Installation

To verify the installation, open Python and run the following:

```python
import torch_structure as ts
```

If no errors occur you have succesfully installed `torch_structure`.
