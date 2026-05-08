$ErrorActionPreference = "Stop"

if ($env:CI_PROJECT_DIR) { Set-Location $env:CI_PROJECT_DIR }

if (-not $env:PIXI_ENV) {
  throw "PIXI_ENV is not set."
}

if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
  throw "git-lfs is required for Windows CI to fetch large test fixtures (Git LFS pointers checked out). Install git-lfs in the Windows CI image."
}
git lfs install --local
$LfsSw = [System.Diagnostics.Stopwatch]::StartNew()
git lfs pull
$LfsSw.Stop()
Write-Host ("git lfs pull took {0:N1}s" -f $LfsSw.Elapsed.TotalSeconds)

$PixiEnvDir = "C:\work\.pixi\envs\$env:PIXI_ENV"
$PythonExe = Join-Path $PixiEnvDir "python.exe"
if (-not (Test-Path $PythonExe)) {
  throw "Expected pre-baked Python not found for $env:PIXI_ENV at $PythonExe"
}

if ("$env:CI_COMMIT_REF_PROTECTED" -ne "true") {
  # Compare against protected baseline to decide how to refresh the environment.
  # This captures all relevant branch changes, not only the latest push.
  git fetch origin main --quiet
  $ChangedFiles = (git diff --name-only origin/main...HEAD)
  if ($LASTEXITCODE -ne 0) { throw "Failed to diff against origin/main...HEAD" }

  $PixiLockChanged = $ChangedFiles -contains "pixi.lock"
  $PyprojectChanged = $ChangedFiles -contains "pyproject.toml"

  Push-Location "C:\work"
  if ($PixiLockChanged) {
    Write-Host "Unprotected ref: pixi.lock changed -> running frozen pixi install for $env:PIXI_ENV"
    pixi install --frozen -e $env:PIXI_ENV
    if ($LASTEXITCODE -ne 0) { throw "pixi install failed for $env:PIXI_ENV" }
  } elseif ($PyprojectChanged) {
    Write-Host "Unprotected ref: pyproject.toml changed without pixi.lock -> running editable pip install for $env:PIXI_ENV"
    # Keep dependency set from the pre-baked Pixi env; refresh only the editable project install.
    & $PythonExe -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw "pip editable install failed for $env:PIXI_ENV" }
  } else {
    Write-Host "Unprotected ref: no pixi.lock/pyproject.toml changes -> using pre-baked $env:PIXI_ENV environment"
  }
  Pop-Location
} else {
  Write-Host "Protected ref: using pre-baked $env:PIXI_ENV environment"
}

$EnvPathEntries = @(
  $PixiEnvDir,
  (Join-Path $PixiEnvDir "Library\mingw-w64\bin"),
  (Join-Path $PixiEnvDir "Library\usr\bin"),
  (Join-Path $PixiEnvDir "Library\bin"),
  (Join-Path $PixiEnvDir "Scripts"),
  (Join-Path $PixiEnvDir "bin")
) | Where-Object { Test-Path $_ }
$env:PATH = ($EnvPathEntries -join ";") + ";" + $env:PATH
$env:CONDA_PREFIX = $PixiEnvDir
$env:CONDA_DEFAULT_ENV = $env:PIXI_ENV
$env:CONDA_SHLVL = "1"

& $PythonExe --version
& $PythonExe -c "import numpy; print('numpy', numpy.__version__)"
& $PythonExe -c "import os,platform; print(platform.platform()); print('cpu', os.cpu_count())"
& $PythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('xdist') else 1)"
$UseXdist = ($LASTEXITCODE -eq 0)
$DurationsArgs = @()
if (("$env:CI_DEBUG" -eq "1") -or ("$env:CI_DEBUG" -eq "true")) {
  $DurationsArgs = @("--durations=25", "--durations-min=1")
}
$PytestArgs = @("-m", "pytest", "--import-mode=importlib") + $DurationsArgs + @("-m", "not local", "tests/")
if ($UseXdist) {
  Write-Host "Using pytest-xdist (-n logical)"
  $PytestArgs = @("-m", "pytest", "-n", "logical", "--dist", "loadfile", "--import-mode=importlib") + $DurationsArgs + @("-m", "not local", "tests/")
} else {
  Write-Host "pytest-xdist not available; running serial pytest"
}
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
& $PythonExe @PytestArgs
$Stopwatch.Stop()
if ($LASTEXITCODE -ne 0) { throw "Tests failed for $env:PIXI_ENV" }
Write-Host ("pytest (env={0}, xdist={1}) took {2:N1}s" -f $env:PIXI_ENV, $UseXdist, $Stopwatch.Elapsed.TotalSeconds)
