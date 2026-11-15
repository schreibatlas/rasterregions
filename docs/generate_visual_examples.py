"""Regenerate or verify the deterministic images used by the examples page."""

from __future__ import annotations

import argparse
import runpy
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DESTINATION = ROOT / "docs" / "_static" / "visual_examples"
OUTPUTS = ROOT / "build" / "examples"

IMAGES = {
    "01_input.png": EXAMPLES / "assets" / "01_multirow_sparse.png",
    "01_output.png": OUTPUTS / "01_auto_regions" / "regions.png",
    "02_input.png": EXAMPLES / "assets" / "02_expected_count_three.png",
    "02_output.png": OUTPUTS / "02_expected_count" / "regions.png",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if committed images are stale")
    args = parser.parse_args()

    runpy.run_path(str(EXAMPLES / "generate_assets.py"), run_name="__main__")
    for name in ("01_auto_regions.py", "02_expected_count.py"):
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")

    if args.check:
        stale = [
            filename
            for filename, source in IMAGES.items()
            if not (DESTINATION / filename).is_file()
            or source.read_bytes() != (DESTINATION / filename).read_bytes()
        ]
        if stale:
            raise SystemExit(f"stale documentation images: {', '.join(stale)}")
        print("documentation images are current")
        return

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, source in IMAGES.items():
        shutil.copyfile(source, DESTINATION / filename)
        print(f"wrote {DESTINATION / filename}")


if __name__ == "__main__":
    main()
