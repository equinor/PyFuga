#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ALLOW_GLOBS = [
    "pyfuga/**",
    "tests/**",
    "docs/**",
    "pyproject.toml",
    "environment.yml",
    "pixi.lock",
    ".coveragerc",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "scripts/dev.py",
]


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def allowed(name: str) -> bool:
    normalized = name.lstrip("./")
    return any(fnmatch.fnmatch(normalized, pat) for pat in ALLOW_GLOBS)


def filter_archive(archive_name: str) -> None:
    archive = Path(archive_name)
    tmp = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(archive, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if not allowed(info.filename):
                continue
            zout.writestr(info, zin.read(info.filename))
    shutil.move(tmp, archive)


def ensure_token(use_sandbox: bool) -> str:
    token = os.environ.get("ZENODO_SANDBOX_TOKEN" if use_sandbox else "ZENODO_PRODUCTION_TOKEN", "")
    if not token:
        print("No Zenodo token configured for selected target.", file=sys.stderr)
        print("Set ZENODO_SANDBOX_TOKEN or ZENODO_PRODUCTION_TOKEN.", file=sys.stderr)
        raise SystemExit(1)
    return token


def select_record(use_sandbox: bool) -> str:
    if use_sandbox:
        return (
            os.environ.get("ZENODO_SANDBOX_RECORD")
            or os.environ.get("ZENODO_RECORD")
            or os.environ.get("ZENODO_RECORD", "")
        )
    return (
        os.environ.get("ZENODO_PRODUCTION_RECORD")
        or os.environ.get("ZENODO_RECORD")
        or os.environ.get("ZENODO_RECORD", "")
    )


def write_metadata(tag: str, metadata_path: str) -> str:
    tag_version = tag[1:] if tag.startswith("v") else tag
    pub_date = output(["git", "show", "-s", "--format=%cs", f"{tag}^{{commit}}"]).splitlines()[0].strip()
    with open(".zenodo.json", encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data.setdefault("metadata", {})
    metadata["version"] = tag_version
    metadata["publication_date"] = pub_date
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return tag_version


def make_archive(tag: str, archive_name: str) -> None:
    run(["git", "archive", "--format", "zip", "--output", archive_name, tag])
    filter_archive(archive_name)


def parse_verify_record(send_output: str, configured_record: str) -> str:
    matches = re.findall(r"record '(\d+)'", send_output)
    if matches:
        return matches[-1]
    return configured_record


def send_to_zenodo(
    token: str,
    use_sandbox: bool,
    auto_publish: bool,
    record: str,
    metadata_path: str,
    archive_name: str,
) -> tuple[str, str]:
    cmd = ["g2z-send"]
    if record:
        cmd += ["-i", record]
    cmd += ["-t", token]
    if use_sandbox:
        cmd += ["-s"]
    if auto_publish:
        cmd += ["-p"]
    cmd += ["-m", metadata_path, archive_name]
    result = run(cmd, check=False, capture=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
    return result.stdout + result.stderr, record


def verify_metadata(record: str, token: str, use_sandbox: bool) -> None:
    if not record:
        return
    base_cmd = ["g2z-get-meta", "-i", record, "-t", token]
    if use_sandbox:
        base_cmd.insert(3, "-s")
    cmd = [*base_cmd, "-d"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        source = "draft"
    except subprocess.CalledProcessError as exc:
        if "status: 404" not in (exc.output or ""):
            raise
        out = subprocess.check_output(base_cmd, text=True)
        source = "record"
    payload = json.loads(out)
    publication_date = payload.get("metadata", {}).get("publication_date")
    print(f"Verified Zenodo {source} metadata for record {record} publication_date = {publication_date}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared Zenodo upload flow for CI jobs.")
    parser.add_argument("--mode", choices=["release", "manual"], required=True)
    args = parser.parse_args()

    if args.mode == "release":
        use_sandbox = env_bool("ZENODO_SANDBOX", False)
        target_tag = os.environ.get("CI_COMMIT_TAG", "")
        if not target_tag:
            print("CI_COMMIT_TAG is required for release mode.", file=sys.stderr)
            return 1
    else:
        use_sandbox = env_bool("ZENODO_USE_SANDBOX", True)
        if os.environ.get("CI_COMMIT_REF_PROTECTED", "false") != "true" and not use_sandbox:
            print("Ref is not protected; refusing production Zenodo upload.", file=sys.stderr)
            print("Set ZENODO_USE_SANDBOX=true, or run from a protected ref.", file=sys.stderr)
            return 1
        run(["git", "fetch", "--tags"], check=False)
        target_tag = os.environ.get("ZENODO_TARGET_TAG") or os.environ.get("CI_COMMIT_TAG", "")
        if not target_tag:
            print("ZENODO_TARGET_TAG is not set (example: v0.1.0)", file=sys.stderr)
            return 1

    token = ensure_token(use_sandbox)
    record = select_record(use_sandbox)
    auto_publish = env_bool("ZENODO_AUTO_PUBLISH", False)

    tag_version = target_tag[1:] if target_tag.startswith("v") else target_tag
    archive_name = f"PyFuga-v{tag_version}.zip"
    metadata_path = ".zenodo.generated.json"

    make_archive(target_tag, archive_name)
    write_metadata(target_tag, metadata_path)

    try:
        send_output, configured_record = send_to_zenodo(
            token=token,
            use_sandbox=use_sandbox,
            auto_publish=auto_publish,
            record=record,
            metadata_path=metadata_path,
            archive_name=archive_name,
        )
    except subprocess.CalledProcessError as exc:
        merged = (exc.output or "") + (exc.stderr or "")
        if (
            args.mode == "manual"
            and use_sandbox
            and record
            and (
                ("status: 404" in merged and "persistent identifier does not exist" in merged)
                or ("status: 410" in merged and "Record deleted" in merged)
            )
        ):
            print(f"Configured sandbox record '{record}' no longer exists; creating a new sandbox record.")
            send_output, configured_record = send_to_zenodo(
                token=token,
                use_sandbox=use_sandbox,
                auto_publish=auto_publish,
                record="",
                metadata_path=metadata_path,
                archive_name=archive_name,
            )
        else:
            return exc.returncode

    verify_record = parse_verify_record(send_output, configured_record)
    verify_metadata(verify_record, token, use_sandbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
