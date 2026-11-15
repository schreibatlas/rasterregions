"""Verify sparse mixed-mask, quantization, and exact-count edge cases."""

from pathlib import Path

import numpy as np

from rasterregions import detect_regions, foreground_mask

ROOT = Path(__file__).resolve().parent


def main() -> None:
    mixed = ROOT / "assets" / "07_mixed_black_color.png"
    mixed_result = detect_regions(mixed, mode="auto")
    assert mixed_result.mask_result.mask[90, 60]
    assert mixed_result.mask_result.mask[90, 195]
    assert mixed_result.mask_result.mask[90, 330]
    assert not mixed_result.mask_result.mask[155, 100]
    assert len(mixed_result.regions) == 3
    print("mixed black/color auto:", mixed_result.mask_result.method, len(mixed_result.regions))

    quantization_fixture = np.full((40, 40, 3), 255, dtype=np.uint8)
    quantization_fixture[10:30, 18:22] = 15
    mask = foreground_mask(quantization_fixture, mode="grayscale")
    assert mask.mask[10:30, 18:22].all()
    assert not mask.mask[0, 0]
    print("grayscale quantization threshold:", mask.threshold)

    touching = ROOT / "assets" / "06_touching_objects.png"
    touching_result = detect_regions(touching, expected_count=3)
    assert touching_result.clean_mask.any()
    assert touching_result.status == "ok"
    assert len(touching_result.regions) == 3
    print("thin-connected expected_count=3:", touching_result.status, len(touching_result.regions))

    single = ROOT / "assets" / "08_single_region.png"
    single_result = detect_regions(single, expected_count=1)
    assert single_result.status == "ok"
    assert len(single_result.regions) == 1
    print("single expected_count=1:", single_result.status, len(single_result.regions))


if __name__ == "__main__":
    main()
