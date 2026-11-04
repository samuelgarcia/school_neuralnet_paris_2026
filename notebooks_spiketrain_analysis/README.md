# Minischool GDR 2026 — Spike Trains

The goal of this repo is to get acquainted to the main methods used to analyze spike trains and their flaws. It won't be possible to cover all the available methods, so opinionated choices have been made. I give below the installation instructions to run the notebooks. We cover 5 topics:

- Spike rate and spike intervals
- Waveform and cell types
- Connectivity / cross correlograms
- Population coding and embedding
- Decoding

The main track of this workshop is using Python. We take advantage of many great python packages that deserve their stars on GitHub:

- [spike interface](https://github.com/SpikeInterface/spikeinterface)
- [pynapple](https://pynapple.org)
- [ZETApy](https://github.com/JorritMontijn/zetapy)
  
If you like Julia like me you might want to try at your own risk the Julia folder and some pilot packages that I am currently developing:

- [Judananas](https://github.com/PierreLeMerre/Judananas.jl)
- [ZETAju](https://github.com/PierreLeMerre/ZetaJu)


# Installation Instructions

## 1. Get the code

```bash
git clone https://github.com/samuelgarcia/school_neuralnet_paris_2026.git
cd school_neuralnet_paris_2026
```

## 2. Python

### Requirements

We recommend the use of [uv](https://docs.astral.sh/uv/) + [VScode](https://code.visualstudio.com/?_exp_home=e855cfac50). 

- [uv](https://docs.astral.sh/uv/), which also installs the right Python (3.13) for you:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Install

```bash
cd python
uv sync
```

This creates `python/.venv` with the exact package versions in `uv.lock`
(pynapple, h5py, matplotlib, ipykernel, jupytext).

### Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user --name minischool-py --display-name "Python (minischool)"
```

The notebooks then show up with the **Python (minischool)** kernel.

### Check

```bash
uv run python -c "import pynapple; print(pynapple.__version__)"
```

### Run

- Script: `uv run python test.py`
- Notebook: open `python/01_load_spikes_python.ipynb` and select the **Python (minischool)** kernel.

Always use `uv run ...` or the **Python (minischool)** kernel. Another Python (system, Homebrew, another venv)
will not have pynapple and fails with `ModuleNotFoundError: No module named 'pynapple'`.


## 3. Julia

### Requirements

- Julia 1.12, installed with [juliaup](https://github.com/JuliaLang/juliaup):

  ```bash
  curl -fsSL https://install.julialang.org | sh
  juliaup add 1.12
  juliaup default 1.12
  ```

### Install

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the exact versions in `Manifest.toml` (HDF5, Plots), and Judananas from GitHub.
The first time takes a few minutes (download and precompilation).

### Register the Jupyter kernel

```bash
julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Julia")'
```

The notebooks then show up with the **Julia 1.12** kernel.

### Check

```bash
julia --project=. -e 'using Judananas; println(pathof(Judananas))'
```

### Run

- Script: `julia -i test.jl` (from `julia/`)
- Notebook: open `julia/01_load_spikes_julia.ipynb` and select the **Julia 1.12** kernel.

To get the latest Judananas from GitHub later: `julia --project=. -e 'using Pkg; Pkg.update("Judananas")'`.

## 4. Open the notebooks

Any of these works with the kernels above:

- **VS Code** with the Python, Jupyter and Julia extensions.
- **JupyterLab**, from the repo root:

  ```bash
  uv run --project python --with jupyterlab jupyter lab
  ```

- **Zed** for the `# %%` scripts (`test.py`, `test.jl`): the `.zed/` settings select both kernels
  for the REPL (run a cell with Cmd+Shift+Enter).

## 5. Data

The NWB file is not in the repo. The notebooks and scripts read it from:

```
/Volumes/MorseSSD/NPX_Database/frozen_Aversion_Nat_Submission/216301_20200521-probe0.nwb
```

If the file is somewhere else, change the `NWB = ...` line at the top of each notebook and script.
