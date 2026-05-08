# GitLab CI notes

## CI images: which tags matter

- **Linux** jobs use **`ci/python-*-uv`**, **`ci/python-*-uv-tests`**, and **`ci/python-*-uv-docs`** with the **`:latest`** tag (see `common.yml`, `test.yml`, and docs jobs in `build.yml`).
- **Windows** jobs use **`ci/windows-pixi-*`** images with tag **`${ENV_TAG}`**, a short content-derived hash from `ci/Dockerfile.windows-pixi-base`, `ci/Dockerfile.windows-pixi-ci-env`, and `pixi.lock`. It is computed in **`prepare_windows_env_tag`** and passed to test jobs via dotenv artifacts. On **protected** refs the tag reflects those inputs; on **unprotected** refs it is **`latest`** so pipelines pull an existing registry image (see comments in `build.yml`).

## Registry storage vs runner disk

- **GitLab Container Registry** cleanup policies reduce registry object count and storage; they do not free space on a shell executor’s local Docker store.
- **Windows image build** jobs in `build.yml` remove obsolete **named** tags under `windows-pixi*` (not `:<none>` — those are dangling until prune). Job **`prune_windows_docker_images`** runs once after **`build_windows_pixi_base`** (optional) and all **`build_windows_pixi_envs`** matrix jobs, removes **dangling** images whose repository path starts with **`CI_REGISTRY_IMAGE`** (so other projects on the same Docker host are not affected), then **`docker system df`**.

## After images were removed locally or in the registry

1. Ensure **`ci/windows-pixi-base`** and **`ci/windows-pixi-py*-ci`** (and matrix envs) exist in the registry with **`:latest`** and the current **content hash** tag if Windows tests should use the pre-baked Pixi env without reinstalling everything from scratch.
2. On a **protected** branch, run a pipeline that triggers **`build_windows_pixi_base`** and **`build_windows_pixi_envs`** (change the paths listed under those jobs’ `rules`, or use a manual run with **`FORCE_WINDOWS_IMAGE_BUILD=1`** where allowed).
3. Linux test images: run **`build_linux_test_images`** (or **`FORCE_LINUX_TEST_IMAGE_BUILD=1`**) so **`:latest`** is republished if needed.

## Layout

- [`.gitlab-ci.yml`](../.gitlab-ci.yml) — stages and includes.
- [`common.yml`](common.yml) — shared templates and defaults.
- [`build.yml`](build.yml) — image builds and related build stage jobs.
- [`test.yml`](test.yml) — pytest and lint jobs.
- [`deploy.yml`](deploy.yml) — deploy stage.

## CI structure notes

- Shared rule templates live in `common.yml` (for example `.rules_on_code_changes` and `.rules_python_matrix_on_changes_or_schedule`) and are the preferred source for trigger semantics.
- `build.yml` uses YAML anchors for repeated image-build inputs (Kaniko registry auth and Windows image-change path sets) to keep repeated job blocks consistent.

## Planned script extraction points

- Windows pytest job logic in `test.yml` (`test_windows_default` and `test_windows_matrix`) is intentionally targeted for extraction to `ci/scripts/` in a follow-up pass.
- Zenodo publish/upload logic in `deploy.yml` (`publish_zenodo` and `zenodo_upload_for_tag`) is also targeted for extraction to shared helpers under `ci/scripts/`.
