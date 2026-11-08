# Installation Guide

Welcome to the installation guide for `torch-structure`. To ensure a smooth installation process, please follow the steps below.

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
uv sync
```

### 2. (Optional) Install a specific PyTorch build

PyTorch is installed automatically as a dependency of `torch-structure`, using the default build from PyPI: CPU-only on Windows and macOS, CUDA-enabled on Linux.

If you need a different build (e.g. CUDA on Windows, or a specific CUDA version), install PyTorch first using the command for your system from the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/). For example, for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`torch-structure` will then use the already-installed PyTorch.


### 3. Install `torch-structure`

Install the `torch-structure` package via pip:

```bash
pip install torch-structure
```

!!! info "Alternative: Install directly from GitHub"
    To install the very latest version of `torch-structure`, clone the [GitHub repository](https://github.com/lazlo-bleker/torch-structure):

    ```bash
    git clone https://github.com/lazlo-bleker/torch-structure.git
    ```

    Followed by creating an editable installation using the following command from within the cloned repository:

    ```bash
    pip install -e .  
    ```

!!! tip "Optional: PyTorch Geometric extensions"
    `torch-structure` does not require the compiled PyTorch Geometric extensions (`torch_scatter`, `pyg_lib`, ...). If they are installed, PyTorch Geometric uses them automatically where they are faster. See the [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for how to install them.

### 4. Verify the Installation

To verify the installation, open Python and run the following:

```python
import torch_structure as ts
```

If no errors occur you have succesfully installed `torch-structure`.
