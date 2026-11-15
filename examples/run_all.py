"""Generate fixtures and run the canonical rasterregions example bundle."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLE_SCRIPTS = (
    "00_quickstart.py",
    "01_auto_regions.py",
    "02_expected_count.py",
    "03_color_mask.py",
    "04_dark_paper.py",
    "05_component_isolation.py",
    "06_numpy_input.py",
    "07_edge_cases.py",
)


def main() -> None:
    runpy.run_path(str(ROOT / "generate_assets.py"), run_name="__main__")
    for name in EXAMPLE_SCRIPTS:
        path = ROOT / name
        print(f"\n=== {name} ===")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
