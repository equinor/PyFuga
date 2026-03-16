from __future__ import annotations

import argparse
import difflib
import hashlib
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
RECIPE = ROOT / "conda-recipe" / "meta.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_line(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for pattern: {pattern!r}, got {count}")
    return updated


def canonical_dep(dep: str) -> str:
    dep = dep.strip()
    dep = re.sub(r"\s+", "", dep)
    match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", dep)
    if not match:
        return dep.lower()
    return f"{match.group(1).lower()}{match.group(2)}"


def parse_recipe_run_requirements(recipe_text: str) -> list[str]:
    return parse_recipe_requirements(recipe_text, "run")


def parse_recipe_requirements(recipe_text: str, section: str) -> list[str]:
    requirements_match = re.search(r"^requirements:\n(?P<body>(?:^  .*\n?)*)", recipe_text, flags=re.MULTILINE)
    if not requirements_match:
        raise RuntimeError("Could not parse requirements section from conda-recipe/meta.yaml")

    requirements_body = requirements_match.group("body")
    section_match = re.search(rf"^  {section}:\n(?P<body>(?:^    - .*\n?)*)", requirements_body, flags=re.MULTILINE)
    if not section_match:
        raise RuntimeError(f"Could not parse requirements.{section} from conda-recipe/meta.yaml")

    run_lines = [line.strip() for line in section_match.group("body").splitlines() if line.strip()]
    run_deps = [line[len("- ") :].strip() for line in run_lines if line.startswith("-")]

    if not run_deps:
        raise RuntimeError(f"Could not parse requirements.{section} from conda-recipe/meta.yaml")

    return run_deps


def parse_test_imports(recipe_text: str) -> list[str]:
    test_match = re.search(r"^test:\n(?P<body>(?:^  .*\n?)*)", recipe_text, flags=re.MULTILINE)
    if not test_match:
        raise RuntimeError("Could not parse test section from conda-recipe/meta.yaml")
    imports_match = re.search(r"^  imports:\n(?P<body>(?:^    - .*\n?)*)", test_match.group("body"), flags=re.MULTILINE)
    if not imports_match:
        raise RuntimeError("Could not parse test.imports from conda-recipe/meta.yaml")

    imports = [line.strip()[len("- ") :] for line in imports_match.group("body").splitlines() if line.strip()]
    if not imports:
        raise RuntimeError("Could not parse test.imports from conda-recipe/meta.yaml")
    return imports


def get_scalar(recipe_text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, recipe_text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {label} from conda-recipe/meta.yaml")
    return match.group("value").strip()


def parse_recipe_maintainers(recipe_text: str) -> list[str]:
    extra_match = re.search(r"^extra:\n(?P<body>(?:^  .*\n?)*)", recipe_text, flags=re.MULTILINE)
    if not extra_match:
        return []
    maintainers_match = re.search(
        r"^  recipe-maintainers:\n(?P<body>(?:^    - .*\n?)*)", extra_match.group("body"), flags=re.MULTILINE
    )
    if not maintainers_match:
        return []

    maintainers = [line.strip()[len("- ") :] for line in maintainers_match.group("body").splitlines() if line.strip()]
    return maintainers


def expected_run_requirements(pyproject_data: dict) -> list[str]:
    project = pyproject_data["project"]
    requires_python = project["requires-python"]
    dependencies = project["dependencies"]
    return [f"python {requires_python}", *dependencies]


def expected_host_requirements(pyproject_data: dict) -> list[str]:
    project = pyproject_data["project"]
    build_system_requires = pyproject_data["build-system"]["requires"]
    return [f"python {project['requires-python']}", "pip", *build_system_requires]


def compare_dependency_sets(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    expected_set = {canonical_dep(dep) for dep in expected}
    actual_set = {canonical_dep(dep) for dep in actual}
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return missing, extra


def verify_remote_source_checksum(url: str, expected_sha256: str, timeout: float = 20.0) -> tuple[bool, str]:
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=timeout) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except URLError as error:
        return False, f"Failed to download source.url for checksum verification: {error}"

    actual = digest.hexdigest()
    if actual != expected_sha256:
        return False, f"source.url checksum mismatch: expected {expected_sha256}, got {actual}"
    return True, ""


def render_synced_recipe(recipe_text: str, name: str, version: str, source_url: str, checksum: str) -> str:
    updated = recipe_text
    updated = replace_line(updated, r"^  name:\s*.*$", f"  name: {name}")
    updated = replace_line(updated, r"^  version:\s*.*$", f"  version: {version}")
    updated = replace_line(updated, r"^  url:\s*.*$", f"  url: {source_url}")
    updated = replace_line(updated, r"^  sha256:\s*.*$", f'  sha256: "{checksum}"')
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync conda-recipe/meta.yaml package version and source checksum with pyproject + built sdist."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not modify files; fail if conda-recipe/meta.yaml is out of sync.",
    )
    parser.add_argument(
        "--check-remote-source",
        action="store_true",
        help="Also download source.url and verify its sha256 (use after artifact is available at source.url).",
    )
    args = parser.parse_args()

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    name = data["project"]["name"]
    version = data["project"]["version"]

    sdist = ROOT / "dist" / f"{name}-{version}.tar.gz"
    if not sdist.exists():
        build_cmd = f"pixi run bash -lc 'cd /tmp && python -m build --sdist {ROOT}'"
        raise FileNotFoundError(f"Missing sdist: {sdist}. Build it first with: {build_cmd}")

    checksum = sha256_file(sdist)
    source_url = f"https://pypi.io/packages/source/{name[0]}/{name}/{name}-{version}.tar.gz"

    recipe = RECIPE.read_text(encoding="utf-8")
    synced_recipe = render_synced_recipe(recipe, name, version, source_url, checksum)

    expected_run = expected_run_requirements(data)
    expected_host = expected_host_requirements(data)
    actual_host = parse_recipe_requirements(recipe, "host")
    actual_run = parse_recipe_run_requirements(recipe)
    missing_host_deps, extra_host_deps = compare_dependency_sets(expected_host, actual_host)
    missing_run_deps, extra_run_deps = compare_dependency_sets(expected_run, actual_run)

    if missing_host_deps or extra_host_deps:
        print("conda-recipe/meta.yaml requirements.host is out of sync with pyproject.toml/build-system")
        if missing_host_deps:
            print("Missing in recipe host requirements:")
            for dep in missing_host_deps:
                print(f"  - {dep}")
        if extra_host_deps:
            print("Extra in recipe host requirements:")
            for dep in extra_host_deps:
                print(f"  - {dep}")
        return 1

    if missing_run_deps or extra_run_deps:
        print("conda-recipe/meta.yaml requirements.run is out of sync with pyproject.toml")
        if missing_run_deps:
            print("Missing in recipe run requirements:")
            for dep in missing_run_deps:
                print(f"  - {dep}")
        if extra_run_deps:
            print("Extra in recipe run requirements:")
            for dep in extra_run_deps:
                print(f"  - {dep}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    recipe_name = get_scalar(recipe, r"^package:\n(?:^  .*\n)*?^  name:\s*(?P<value>.+)$", "package.name")
    recipe_version = get_scalar(recipe, r"^package:\n(?:^  .*\n)*?^  version:\s*(?P<value>.+)$", "package.version")
    recipe_source_url = get_scalar(recipe, r"^source:\n(?:^  .*\n)*?^  url:\s*(?P<value>.+)$", "source.url")
    recipe_sha256 = get_scalar(
        recipe,
        r"^source:\n(?:^  .*\n)*?^  sha256:\s*(?P<value>.+)$",
        "source.sha256",
    ).strip('"')
    noarch = get_scalar(recipe, r"^build:\n(?:^  .*\n)*?^  noarch:\s*(?P<value>.+)$", "build.noarch")
    build_number = get_scalar(recipe, r"^build:\n(?:^  .*\n)*?^  number:\s*(?P<value>.+)$", "build.number")
    build_script = get_scalar(recipe, r"^build:\n(?:^  .*\n)*?^  script:\s*(?P<value>.+)$", "build.script")
    about_license = get_scalar(recipe, r"^about:\n(?:^  .*\n)*?^  license:\s*(?P<value>.+)$", "about.license")
    license_file = get_scalar(recipe, r"^about:\n(?:^  .*\n)*?^  license_file:\s*(?P<value>.+)$", "about.license_file")

    if recipe_name != name:
        errors.append(f"package.name mismatch: recipe={recipe_name!r}, pyproject={name!r}")
    if recipe_version != version:
        errors.append(f"package.version mismatch: recipe={recipe_version!r}, pyproject={version!r}")
    if recipe_source_url != source_url:
        errors.append(f"source.url mismatch: recipe={recipe_source_url!r}, expected={source_url!r}")
    if recipe_sha256 != checksum:
        errors.append(f"source.sha256 mismatch: recipe={recipe_sha256!r}, expected={checksum!r}")

    if noarch != "python":
        errors.append(f"build.noarch must be 'python', got {noarch!r}")

    if not re.fullmatch(r"\d+", build_number):
        errors.append(f"build.number must be a non-negative integer, got {build_number!r}")

    if "pip install ." not in build_script:
        errors.append("build.script must install the package via pip (expected 'pip install .')")

    expected_license = str(data["project"]["license"])
    if about_license != expected_license:
        errors.append(f"about.license mismatch: recipe={about_license!r}, pyproject={expected_license!r}")

    if not (ROOT / license_file).exists():
        errors.append(f"about.license_file does not exist: {license_file}")

    test_imports = set(parse_test_imports(recipe))
    required_imports = {"pyfuga", "pyfuga.preluts_generator"}
    missing_imports = sorted(required_imports - test_imports)
    if missing_imports:
        errors.append(f"test.imports missing required entries: {', '.join(missing_imports)}")

    maintainers = parse_recipe_maintainers(recipe)
    if not maintainers:
        errors.append("extra.recipe-maintainers is missing or empty")
    elif any("REPLACE_WITH" in maintainer for maintainer in maintainers):
        warnings.append("extra.recipe-maintainers still contains placeholder value(s)")

    if args.check_remote_source:
        ok, msg = verify_remote_source_checksum(recipe_source_url, recipe_sha256)
        if not ok:
            errors.append(msg)

    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    if args.check:
        if recipe != synced_recipe:
            diff = "\n".join(
                difflib.unified_diff(
                    recipe.splitlines(),
                    synced_recipe.splitlines(),
                    fromfile="meta.yaml (current)",
                    tofile="meta.yaml (expected)",
                    lineterm="",
                )
            )
            print("conda-recipe/meta.yaml is out of sync.")
            print("Run: pixi run sync-conda-recipe")
            if diff:
                print(diff)
            return 1
        print("conda-recipe/meta.yaml is in sync")
        return 0

    RECIPE.write_text(synced_recipe, encoding="utf-8")
    print(f"Updated {RECIPE}")
    print(f"  version: {version}")
    print(f"  source:  {source_url}")
    print(f"  sha256:  {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
