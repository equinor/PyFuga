# PyFuga

[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)
[![documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://topfarm.pages.windenergy.dtu.dk/cuttingedge/pywake/fuga/PyFuga/)


## Overview
**PyFuga** is the Python implementation of the **Fuga** look-up table (LUT) generator previously computed by Preludium and Trafalgar. Fuga is a fast linearised CFD model for predicting the wind-turbine wake fields and wake interactions in wind farms. Fuga solves the linearised RANS equations in a **mixed-spectral formulation**, using a modified "chasing method" to integrate a six-component ODE system. It produces look-up tables that enable efficient and accurate reconstruction of wake fields.

PyFuga is part of the [**PyWake**](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake) ecosystem and provides the LUTs required to run PyWake's Fuga wake deficit model.

### Relationship to yaw modelling

PyFuga computes look-up tables for a single, unyawed turbine only. Yaw-induced wake deflection is handled in PyWake. For implementation details, see PyWake's [FugaDeflection](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake) documentation.

See [QuickStart.ipynb](docs/examples/01_QuickStart.ipynb)

## Documentation

Full documentation (work in progress) is available here:

🔗 **https://topfarm.pages.windenergy.dtu.dk/cuttingedge/pywake/fuga/PyFuga/**

The documentation aims to provide:

- a **high-level description** of the theory behind Fuga (with appendices for more detail),  
- the **QuickStart guide**,  
- the **API reference**,  
- a forthcoming **User Guide** (workflow, LUT generation, integration with PyWake),  
- a forthcoming **Developer Guide** (code structure, numerical routines, contributing).

This will be the primary source for understanding how PyFuga works and how to use it effectively.

## Quick start

A minimal example is available in **[QuickStart.ipynb](docs/examples/01_QuickStart.ipynb)**.

## Look-up tables (LUTs)

PyFuga produces several intermediate look-up tables, referred to as **PreLUTs** and **fLUTs**, before generating the final look-up tables (LUTs) used by PyWake. The final LUTs contain turbine-induced perturbation fields evaluated during wind-farm simulations.

The intermediate stages are internal and only need to be set up when generating LUTs for a new turbine or atmospheric configuration.

## Differences from the legacy Fortran / C++ implementation

### PreLUT generation

- Uses `numpy.linalg.qr` for QR decomposition.

    This mirrors the logic of the original code but not its exact floating-point behaviour, meaning PreLUTs are **not numerically identical** to legacy outputs.

- Removes an additional `h` term in `get_new_h2` present in the Fortran code, which caused unnecessarily small integration steps.

### Trafalgar

- Interpolation changed from linear to cubic.

## Installation

Install via pip:

```bash
pip install pyfuga
```

## Development quickstart
```bash
git clone git@gitlab.windenergy.dtu.dk:TOPFARM/cuttingedge/pywake/fuga/pyfuga.git
cd pyfuga
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

### Windows (Conda)

Create and activate an environment first:
```powershell
conda create -n pyfuga python=3.12 -y
conda activate pyfuga
```

Then follow the development steps above
```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

> Tip: always use `python -m pip ...` to ensure you install into the active environment.

Alternatively, you can create the environment from `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate pyfuga
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Usage

A full usage guide will be added in the documentation.

For now, please refer to [**QuickStart.ipynb**](docs/examples/01_QuickStart.ipynb).

## Contributing

1. Install dev dependencies: `python -m pip install -e ".[dev]"`
2. Install hooks once: `pre-commit install`
3. Before pushing: `pre-commit run --all-files` and `pytest`

If CI fails on formatting, run `pre-commit run --all-files`, commit the changes, and push again.

### Code style

- **Black** for formatting (line length: 120).
- **Ruff** for linting and import sorting (`pyproject.toml` configuration; `pyfuga` is first-party).

### Tests

```bash
pytest
```

### Common commands
```bash
pre-commit run --all-files
pytest
```

### Optional helper
```bash
python scripts/dev.py doctor
python scripts/dev.py fmt
python scripts/dev.py test
```

The `doctor` command prints information about your Python environment and git hooks, which can help diagnose setup issues.

> For more details, see [CONTRIBUTING.md](CONTRIBUTING.md)

## Support

Issues and feature requests can be submitted through the project's GitLab issue tracker.

## Authors and acknowledgements

PyFuga is developed at **DTU Wind and Energy Systems** and builds on the Fuga model described in the technical report by Søren Ott, Mads Mølgaard Pedersen, Gunnar Chr. Larsen, Leonardo Alcayaga, Nils Gaukroger, Elvira Jarmbæk Jacobsen, and colleagues.

The PyFuga project would like to acknowledge **Equinor ASA** for their support of the project over many years.

## Licence

This project is released under the terms of the licence in the [`LICENSE`](LICENSE) file.

## Project status

Active development as part of the PyWake suite. PyFuga continues to evolve alongside ongoing improvements to the Fuga model and its numerical implementation.
