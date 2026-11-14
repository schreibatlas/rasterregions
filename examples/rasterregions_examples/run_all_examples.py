"""Run all standalone rasterregions examples in this bundle."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in sorted(ROOT.glob("example_*.py")):
    print(f"\n=== {path.name} ===")
    runpy.run_path(str(path), run_name="__main__")
