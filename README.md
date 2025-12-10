# PyFuga


## Description
This is the python version of the FUGA LUT generator (Previuosly computed by Preludium and Trafalgar).


[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/cuttingedge/pywake/fuga/pyfuga/-/commits/main)

## Quick start

See [QuickStart.ipynb](QuickStart.ipynb)

## Difference compared to old fortran / cpp implementation

- Prelut
  - QR decompostion using np.linalg.qr. This method orthonormalize analogous to the fortran implementation, but the 
results are not equal. I.e. the prelut data cannot be compared directly
  - The fortran code has an extra `h`-term in get_new_h2, which makes the step too small. This additional term has been removed
- Trafalgar
  - interpolation changed from linear to cubic 

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

- We use **Black** for formatting.
  - Default line length: 120 characters.
  - Use `#fmt: off` / `#fmt: on` sparingly for genuinely hard-to-read edge cases (such as jitted functions with many inputs).

- We use **Ruff** for linting and import sorting.
  - Configuration is in `pyproject.toml`.
  - Import order follow's Ruff's isort rules with `pyfuga` as first-party code.

- We use [pre-commit](https://pre-commit.com/) to keep imports and formatting consistent.
  ```bash
  pip install pre-commit
  pre-commit install
  ```

  Then `isort` (and other hooks) will run automatically on staged files when you commit. You can also run all hooks manually with:
  ```bash
  pre-commit run --all-files
  ```

- Recommended workflow:
  - Install pre-commit: `pip install pre-commit`.
  - Run `pre-commit install` once in your clone.
  - Optionally, run `pre-commit run --all-files` before large refactors.

- Type checking and editor diagnostics still use **Pylance** in VS Code.

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
