# History Rewrite to LFS Runbook (Phases 1-3)

This runbook captures the executed work for the one-time history rewrite project.
It is written to support safe progression to later phases.

## Goal

Prepare for and validate a safe Git history rewrite by:

- capturing measurable baseline metrics,
- freezing migration scope,
- defining backup/rollback safety points,
- running and validating a dry-run rewrite before any production ref changes.

## Baseline Captured (2026-05-05)

Repository object metrics (`git count-objects -vH`):

- `count`: 264
- `size`: 1.15 GiB
- `in-pack`: 3314
- `packs`: 1
- `size-pack`: 137.35 MiB
- `garbage`: 0

LFS runtime baseline:

- `git-lfs` version: `3.4.1`
- Git version: `2.43.0`
- Endpoint: `git@gitlab.windenergy.dtu.dk:TOPFARM/PyFuga.git`

Current checkout LFS footprint:

- `git lfs ls-files` count: **153** tracked files

## Historical Blob Inventory (Current History)

Largest historical blobs include duplicated path eras (`tests/...`, `fuga/tests/...`, and later
prefix variants discovered during dry run), indicating historical moves/renames still contributing
to repository weight.

Top historical blobs (across all refs):

- `tests/test_files/preLUTs_Zeta0=0.00E+00_2_5.nc` ~83.13 MiB
- `fuga/tests/test_files/preLUTs_Zeta0=0.00E+00_2_5.nc` ~83.13 MiB
- `tests/test_files/preLUTs_Zeta0=0.00E+00_1_2.nc` ~12.91 MiB
- `fuga/tests/test_files/preLUTs_Zeta0=0.00E+00_1_2.nc` ~12.91 MiB
- `tests/test_files/preLUTs_Zeta0=-1.00E+00_1_2.nc` ~12.78 MiB
- `fuga/tests/test_files/preLUTs_Zeta0=-1.00E+00_1_2.nc` ~12.78 MiB

Scope-specific summary from dry-run baseline report:

- scoped unique blob payload: **378.18 MiB** across **484** blobs

## Scope Freeze for Rewrite

Primary migration scope (historical Git blobs to LFS):

- `tests/test_files/**/*.nc`
- `tests/test_files/**/*.pre`
- `fuga/tests/test_files/**/*.nc` (legacy path era)
- `fuga/tests/test_files/**/*.pre` (legacy path era)
- `pyfuga/tests/test_files/**/*.nc` (later prefix era)
- `pyfuga/tests/test_files/**/*.pre` (later prefix era)
- `pypreludium/tests/test_files/**/*.nc` (legacy/package split era)
- `pypreludium/tests/test_files/**/*.pre` (legacy/package split era)

## Backup and Rollback Safety Points

Before any rewrite command is run:

1. Create and retain a **server-side snapshot** (mirror/backup project or export).
2. Create a **local mirror backup**:
   - `git clone --mirror <repo-url> pyfuga-pre-rewrite-backup.git`
3. Save immutable backup metadata:
   - default branch tip SHA
   - all release tag SHAs
   - backup creation timestamp and owner
4. Store backup access details in the migration issue/runbook.

GitLab backup push guidance:

- Avoid `git push --mirror` for backup projects on GitLab, because it attempts to push
  hidden refs (for example `refs/merge-requests/*`) that are rejected.
- Push branches and tags only:
  - `git push --prune backup 'refs/heads/*:refs/heads/*' 'refs/tags/*:refs/tags/*'`
- Ensure all LFS objects are present before branch/tag push:
  - `git lfs fetch --all origin`
  - `git lfs push --all backup`
- If LFS reports missing source object OIDs, stop and restore source LFS integrity before Phase 2.

## Snapshot Lock-Down (Server-Side Backup Project)

Immediately after snapshot creation, lock down the backup project to make it effectively immutable.

Project used:

- `git@gitlab.windenergy.dtu.dk:njoga/pyfuga-pre-rewrite-backup.git`

Required hardening steps (GitLab UI):

1. Set project visibility to `Private` (or stricter org policy equivalent).
2. Protect all snapshot branches (at minimum `main`):
   - Allowed to push: `No one` (or `Maintainers` if `No one` is not available).
   - Allowed to merge: `No one` (or `Maintainers` if required by instance policy).
3. Protect all tags:
   - Rule: `*`
   - Allowed to create: `No one` (or `Maintainers` per policy).
4. Limit project members to required maintainers/owners only.
5. Add explicit project description:
   - `Immutable pre-history-rewrite snapshot of TOPFARM/PyFuga created 2026-05-05. Do not develop here. Restore/reference only.`
6. Optionally disable non-essential features (MRs/issues/wiki/packages) for this backup project.

Snapshot verification checklist:

- `git ls-remote --heads backup` returns expected branch refs.
- `git ls-remote --tags backup` returns expected release tags.
- Branch/tag counts match origin for in-scope refs.
- Recorded metadata includes timestamp, owner, default branch SHA, and release tag SHAs.

Rollback principle:

- If production rewrite introduces issues, restore refs from pre-rewrite mirror backup
  and communicate a temporary freeze while restoration completes.

## Communication and Freeze Checklist

Before production cutover:

- [ ] Announce migration project, rationale, and expected impact window.
- [ ] Announce temporary merge freeze for protected branches during cutover.
- [ ] Share contributor warning that local branches will need reset/reclone later.
- [ ] Confirm at least one maintainer on Linux and one on Windows are available for post-rewrite validation.
- [ ] Confirm CI maintainers are available to clear caches/retry pipelines if required.

## Phase 1 Exit Criteria (Go/No-Go to Phase 2)

Proceed to dry run only if all are true:

- [x] Blob inventory report is captured and reviewed.
- [x] Migration scope is frozen and approved.
- [x] Backup and restore points are created and verified.
- [x] Backup project is locked down (protected branches/tags, restricted write access, immutable description).
- [x] Maintainer ownership for rewrite execution is assigned.

If any check fails, do not start rewrite rehearsal.

## Phase 2 Dry-Run Conclusion (2026-05-05)

Phase 2 dry run completed successfully in an isolated mirror clone. History was rewritten with
`git lfs migrate import --everything` across all relevant historical fixture path eras
(`tests/test_files`, `fuga/tests/test_files`, `pyfuga/tests/test_files`, `pypreludium/tests/test_files`)
for `.nc` and `.pre`.

Validation results:

- `git fsck --full --strict --no-dangling`: pass
- `git lfs fsck`: pass
- key refs/tags verified (`main`, `v0.1`, `v0.1.0`, `v0.2.0`, `v0.3.0`)

Measured impact:

- scoped test-fixture historical blob payload: **378.18 MiB -> 2.78 MiB**
- packed Git object size after GC: **137.75 MiB -> 4.79 MiB** (~96.5% reduction)

No production refs were modified in this phase.

## Phase 3 Staging Validation Status (2026-05-05)

Rewritten refs and LFS objects were pushed to staging (`njoga/pyfuga-rewriting-staging`), but CI
parity issues were observed due to image-path and trigger differences versus production namespace.

Observed issue:

- jobs attempted to pull runtime images from
  `registry.windenergy.dtu.dk/njoga/pyfuga-rewriting-staging/ci/...`
  where required tags did not exist (manifest not found), particularly for Windows image variants.
- historical tag pipelines in staging did not bootstrap the same image set expected by newer refs.

Mitigations added:

- CI now supports overriding runtime image pull prefix via `PYFUGA_CI_IMAGE_PREFIX`
  (default remains `${CI_REGISTRY_IMAGE}`).
- Windows test jobs now tolerate refs without a merge base against `origin/main`
  by skipping change-diff optimization instead of failing the job early.

Phase 3 interpretation:

- rewrite/LFS mechanics: validated
- full CI parity: partially validated (staging image parity and runner constraints still apply)

Phase 4 prerequisite from Phase 3:

- run one final staging pipeline with seeded images and verify Linux + Windows fixture-dependent jobs.
- confirm no image pull or LFS fetch errors remain before production cutover planning.

## Phase 3 Pass/Fail Matrix (Staging)

Scope refs:

- `main` (or `chore/staging-ci-image-prefix` as the effective CI-fix surrogate for main in staging)
- `v0.3.0`
- `v0.2.0`
- `v0.1.0`
- `pre-trafalgar-tiling`
- `v0.1`

Legend:

- LFS status: `pass`, `fail`, `unknown`, `blocked`
- Overall: `pass`, `partial`, `fail`, `pending`

### `main` / `chore/staging-ci-image-prefix`

- Pipeline status: **passed** (CI-fix branch run completed successfully).
- LFS status: **pass** (no reported LFS fetch/transport errors in successful run).
- Non-LFS failures seen earlier:
  - Windows image availability / bootstrap parity issues (mitigated for CI-fix branch validation).
  - Windows `origin/main...HEAD` merge-base assumption (mitigated in CI branch).
- Interpretation: **pass** for rewrite/LFS validation on the staging CI-fix surrogate of `main`.

### `v0.3.0`

- Pipeline status: **partial**.
- LFS status: **pass on Linux path** (Linux test jobs passed, no reported LFS errors).
- Non-LFS failures:
  - Windows jobs failed due to missing image availability in staging.
- Interpretation: **partial**. Linux-side rewrite/LFS validation succeeded; Windows validation remains blocked by staging image parity.

### `v0.2.0`

- Pipeline status: **failed**.
- LFS status: **no direct LFS error observed** in reported failures.
- Non-LFS failures:
  - Linux tests failed with
    `TypeError: Passing a Dataset as data_vars to the Dataset constructor is not supported. Use ds.copy() to create a copy of a Dataset.`
  - Windows jobs failed due to missing image availability in staging.
- Interpretation: **partial** for rewrite/LFS signal. Failure modes currently match tag/runtime compatibility and staging-image parity issues rather than rewrite-LFS transport failures, but they limit full end-to-end confidence on this tag.

### `v0.1.0`

- Pipeline status: **failed**.
- LFS status: **no direct LFS error observed** in reported failures.
- Non-LFS failures:
  - Linux tests failed with
    `TypeError: Passing a Dataset as data_vars to the Dataset constructor is not supported. Use ds.copy() to create a copy of a Dataset.`
  - Windows jobs failed due to missing image availability in staging.
- Interpretation: **partial** for rewrite/LFS signal. Failures appear tag/runtime specific and staging-image related rather than rewrite-LFS transport failures, but they reduce confidence for full end-to-end validation on this tag.

### `pre-trafalgar-tiling`

- Pipeline status: **failed**.
- LFS status: **no explicit LFS error reported** so far.
- Non-LFS failures:
  - missing runtime images in staging.
- Interpretation: **partial**. Failure mode is expected CI-image parity issue and can obscure deeper validation.

### `v0.1`

- Pipeline status: **blocked/stuck**.
- LFS status: **blocked** (cannot evaluate while pipeline cannot run).
- Non-LFS failures:
  - runner availability/protection mismatch:
    `There are no active runners online, no runners for the protected branch, or no runners that match all of the job's tags: ci-ubuntu`.
- Interpretation: **blocked**. This is infrastructure/runners policy, not a rewrite signal.

### Matrix Summary (Current)

- Confirmed rewrite-specific LFS transport failures in Phase 3 runs: **none reported**.
- Confirmed blockers/noise sources:
  - staging image namespace/bootstrap gaps (Linux/Windows runtime images),
  - Windows merge-base assumption on rewritten/unrelated branch ancestry (mitigated),
  - runner-tag/protected-ref policy mismatch for older tags.
- Decision status: **Phase 3 = pass with documented staging limitations**.
  - Rewrite/LFS behavior is validated on successful CI paths.
  - Remaining failures are attributable to staging Windows image/runners parity and historical tag/runtime issues, not direct rewrite-LFS transport faults.

## Commands Used for Baseline and Validation

```bash
git count-objects -vH
git lfs env
git lfs ls-files
git fsck --full --strict --no-dangling
git lfs fsck
```

Blob inventory script pattern used:

```bash
git rev-list --objects --all
git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)'
```

## Notes

- This runbook now captures both preflight and dry-run completion state.
- Production rewrite/cutover steps remain pending later phases.
