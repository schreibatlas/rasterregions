"""Pass a NumPy image directly instead of a file path."""

from pathlib import Path

import numpy as np
from PIL import Image

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
source = np.asarray(Image.open(ROOT / "assets" / "02_expected_count_three.png"))
out = ROOT / "outputs" / "06_numpy_input"

result = detect_regions(source, expected_count=3)
assert result.status == "ok"
assert len(result.regions) == 3
write_detection(result, out)
print("source_path:", result.source_path)
print("status/count:", result.status, len(result.regions))
