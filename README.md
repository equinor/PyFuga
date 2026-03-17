<p align="center">
  <img width="460" height="300" src="https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga/raw/main/docs/_static/logo_icon_text.svg">
</p>

**PyFuga** is the Python implementation of the **Fuga** look-up table (LUT) generator for wind turbine wakes, previously computed in FORTRAN and with the Windows Fuga GUI.

PyFuga is part of the [**PyWake**](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake) ecosystem and provides the LUTs required to run PyWake's Fuga wake deficit model.

[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)
[![documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://topfarm.pages.windenergy.dtu.dk/cuttingedge/pywake/fuga/PyFuga/)

## Installation

Install via pip:

```bash
pip install pyfuga
```

or from conda-forge with conda:

```bash
conda install conda-forge:pyfuga
```

or Pixi:

```bash
pixi add pyfuga
```

A minimal example to get you started is available in **[the QuickStart Jupyter notebook](docs/examples/01_QuickStart.ipynb)**.

## Documentation

Learn more about PyFuga at the **[official documentation](https://topfarm.pages.windenergy.dtu.dk/cuttingedge/pywake/fuga/PyFuga/)**.

## Release history

See **[CHANGELOG.md](CHANGELOG.md)** for release notes and version history.

## Contributing

Read more at **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Support

Issues and feature requests can be submitted through the project's [**GitLab issue tracker**](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/PyFuga/-/issues).

## Authors and acknowledgements

PyFuga is developed at **DTU Wind and Energy Systems** and builds on the Fuga model described in the technical report by Søren Ott, Mads Mølgaard Pedersen, Gunnar Chr. Larsen, Leonardo Alcayaga, Nils Gaukroger, Elvira Jarmbæk Jacobsen, and colleagues.

The PyFuga project would like to acknowledge **Equinor ASA** for their support of the project over many years.

## Licence

This project is released under the terms of the licence in the [`LICENSE`](LICENSE) file.

## Project status

Active development as part of the PyWake suite. PyFuga continues to evolve alongside ongoing improvements to the Fuga model and its numerical implementation.
