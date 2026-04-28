# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-04-28

Third public beta release focused on clarifying LUT multiprocessing semantics, tightening the public LUT API around CPU selection, and aligning the non-dimensional eddy-viscosity notation used across code and docs.

### Added

- Optional `n_cpu_stage` per-stage overrides in the public LUT API, allowing
  explicit worker counts for `preluts`, `fluts`, and `trafalgar`.

### Changed

- Updated non-dimensional eddy-viscosity notation to use \(K^* = kK/u_*\)
  consistently in both implementation and theory docs.
- Refactored `pyfuga.preluts_generator.matrices` variable naming from `kK`-style
  terms to `K_star`-style terms without changing numerical behavior.
- **Breaking:** `n_cpu` must be `None` or an `int >= 1` across the public LUT API
  and related stages. `None` means use all CPUs (`os.cpu_count()`, at least `1`);
  `1` is serial. Invalid values such as `0` are rejected.
- **Breaking:** default stage behavior in `get_luts` is now stage-aware when
  `n_cpu`/`n_cpu_stage` are omitted: all stages use all CPUs by default.
  Precedence is explicit: stage defaults -> global `n_cpu` (if provided) ->
  `n_cpu_stage` per-stage overrides.

### Fixed

- Multiprocessing mapping is now uniformized across `preluts`, `fluts`, and
  `trafalgar` via a shared helper with consistent spawn-based pool handling.

## [0.2.0] - 2026-04-02

Second public beta release focused on API expansion, LUT pipeline refactoring, and release/distribution workflow improvements.

### Added

- Public `get_fluts()` and `get_preluts()` API wrappers in `pyfuga`, in addition to `get_luts()`.
- New `pyfuga.paths` helper module for stable naming/path construction of PreLUT, fLUT, and LUT artifacts.
- Tests for path-id stability, orchestration, and cache behaviour across LUT generation stages.
- Sphinx multiversion documentation setup with version selector and updated documentation templates.
- Zenodo metadata and automated Zenodo upload integration in release CI.
- Conda packaging/deployment jobs and recipe sync checks in release pipeline.

### Changed

- Refactored LUT orchestration in `pyfuga._fuga` to split responsibilities into:
  - `get_preluts()`
  - `get_fluts()`
  - `get_luts()`
- Moved vertical spacing constant to shared constant `DS` for consistent level/range computations.
- Improved docs build and publishing workflow for branch/tag-aware multiversion output.
- Updated project metadata and release tooling for the `0.2.x` development cycle.
- Migrated large test LUT fixtures to Git LFS and optimised CI checkout behaviour.

### Fixed

- Conda recipe alignment and source checksum verification flow for release artifacts.
- TestPyPI/packaging credential handling and release validation checks in CI.
- Minor documentation and metadata inconsistencies (links, typos, release references).

### Notes

- This release includes significant CI/CD and packaging infrastructure work to support reproducible publishing across PyPI, docs, Zenodo, and conda workflows.
- Conda-forge publication may follow separately from this tag depending on staged-recipes review timing.

## [0.1.0] - 2026-03-17

Initial public beta release of PyFuga.

### Added

- Public Python API for Fuga look-up table (LUT) generation
- Three-stage pipeline: PreLUT generation, Fourier LUT, Trafalgar output
- Support for Python 3.10–3.14
- Comprehensive documentation with theory and examples
- CI/CD automation for testing and publishing

### Changed

- Maturity classification: Alpha -> Beta

### Notes

- This is the first public release. API may still evolve based on user feedback.
