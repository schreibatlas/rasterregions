"""Pass a NumPy image directly instead of a file path."""

from pathlib import Path

import numpy as np
from PIL import Image

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parents[1] / "build" / "rasterregions_examples"


def main() -> None:
    with Image.open(ROOT / "assets" / "02_expected_count_three.png") as image:
        source = np.asarray(image)
    result = detect_regions(source, expected_count=3)
    assert result.status == "ok"
    assert len(result.regions) == 3
    write_detection(result, BUILD / "06_numpy_input")
    print("source_path:", result.source_path)
    print("status/count:", result.status, len(result.regions))


if __name__ == "__main__":
    main()
