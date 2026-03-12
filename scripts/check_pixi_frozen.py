from __future__ import annotations

import subprocess
import sys


def main() -> int:
    envs = ["py310", "py311", "py312", "py313", "py314"]

    print("Checking pixi frozen installs for environments:", ", ".join(envs))

    for env in envs:
        print(f"\n=== pixi install --frozen -e {env} ===")
        result = subprocess.run(["pixi", "install", "--frozen", "-e", env])
        if result.returncode != 0:
            print(
                f"\nERROR: pixi.lock is not in sync for environment '{env}'.\n"
                "Run 'pixi lock' locally, commit the updated pixi.lock, and push again."
            )
            return result.returncode

    print("\nAll Pixi environments are in sync with pixi.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
