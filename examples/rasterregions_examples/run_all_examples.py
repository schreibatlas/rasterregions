"""Generate fixtures and run all standalone rasterregions examples."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    runpy.run_path(str(ROOT / "generate_assets.py"), run_name="__main__")
    for path in sorted(ROOT.glob("example_*.py")):
        print(f"\n=== {path.name} ===")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
