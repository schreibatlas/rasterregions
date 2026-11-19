"""Run deterministic, opt-in rasterregions performance workloads."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import detect_spaced_regions, foreground_evidence, foreground_mask
from rasterregions._partition import PartitionResult, _candidate_cut_positions, _partition_row
from rasterregions.components import components
from rasterregions.models import Component, PartitionConfig

T = TypeVar("T")


def _measure(operation: Callable[[], T], *, repeat: int, warmup: int) -> tuple[T, list[float]]:
    result: T
    for _ in range(warmup):
        result = operation()
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
    return result, samples


def _component_cases() -> list[tuple[str, np.ndarray]]:
    size = 512
    dense = np.ones((size, size), dtype=bool)
    sparse = np.zeros((size, size), dtype=bool)
    for y in range(0, size, 16):
        for x in range(0, size, 16):
            sparse[y : y + 3, x : x + 3] = True
    return [("components_dense", dense), ("components_sparse", sparse)]


def _partition_case(width: int) -> tuple[np.ndarray, tuple[Component, ...], PartitionConfig]:
    height = 96
    expected_count = 8
    row_mask = np.zeros((height, width), dtype=bool)
    pitch = width // expected_count
    glyph_width = width // 12
    for index in range(expected_count):
        left = index * pitch + (pitch - glyph_width) // 2
        for local_x in range(glyph_width):
            profile = 26 + abs((local_x + index * 7) % 50 - 25)
            top = (height - profile) // 2
            row_mask[top : top + profile, left + local_x] = True
    found = components(row_mask)
    if len(found) != expected_count:
        raise RuntimeError(f"Expected {expected_count} synthetic components, got {len(found)}")
    return row_mask, found, PartitionConfig()


def _partition_operation(
    row_mask: np.ndarray,
    found: tuple[Component, ...],
    config: PartitionConfig,
) -> Callable[[], PartitionResult]:
    def run() -> PartitionResult:
        result = _partition_row(
            row_mask,
            8,
            found,
            x_offset=0,
            config=config,
        )
        if result is None:
            raise RuntimeError("Synthetic row could not be partitioned")
        return result

    return run


def _handwritten_sheet() -> Image.Image:
    row_counts = (12, 8, 6, 11, 11, 5, 10)
    image = Image.new("RGB", (1081, 1198), "white")
    draw = ImageDraw.Draw(image)
    for row_index, count in enumerate(row_counts):
        spacing = (image.width - 150) / (count - 1)
        y = 35 + row_index * 155
        for index in range(count):
            x = 30 + index * spacing
            stroke_width = 86 + (index * 7) % 12
            draw.line(
                (round(x), y + 55, round(x + stroke_width), y + 5),
                fill="black",
                width=4,
            )
            if index % 8 == 0:
                dot_x = round(x + stroke_width - 20)
                draw.ellipse((dot_x - 1, y - 3, dot_x + 1, y - 1), fill="black")
    draw.point((1065, 15), fill="black")
    draw.rectangle((1058, 1110, 1059, 1111), fill="black")
    return image


def _foreground_cases() -> list[tuple[str, np.ndarray]]:
    page = Image.new("L", (1650, 2200), 255)
    draw = ImageDraw.Draw(page)
    for row_index, y in enumerate(range(70, 2120, 55)):
        x = 60
        mark_index = 0
        while x < 1590:
            width = 8 + (row_index * 13 + mark_index * 7) % 32
            draw.line((x, y, x + width, y + 4), fill=0, width=3)
            x += width + 12 + (mark_index * 5) % 17
            mark_index += 1
    draw.line((780, 980, 820, 915), fill=0, width=8)
    draw.ellipse((810, 920, 838, 948), fill=0)
    pixels = np.asarray(page)
    glyph_crop = np.array(pixels[880:1008, 760:888], copy=True)
    return [
        ("foreground_full_page", np.array(pixels, copy=True)),
        ("foreground_cropped_glyph", glyph_crop),
    ]


def run_benchmarks(*, repeat: int, warmup: int) -> dict[str, object]:
    workloads: list[dict[str, object]] = []
    for name, mask in _component_cases():
        found, samples = _measure(partial(components, mask), repeat=repeat, warmup=warmup)
        workloads.append(
            {
                "name": name,
                "shape": list(mask.shape),
                "foreground_fraction": float(np.mean(mask)),
                "component_count": len(found),
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
            }
        )

    for width in (2000, 4000):
        row_mask, found, config = _partition_case(width)
        operation = _partition_operation(row_mask, found, config)
        partition, samples = _measure(operation, repeat=repeat, warmup=warmup)
        projection = np.asarray(row_mask, dtype=np.uint8).sum(axis=0)
        candidate_count = len(_candidate_cut_positions(projection, config))
        workloads.append(
            {
                "name": "partition_row",
                "shape": list(row_mask.shape),
                "width": width,
                "foreground_fraction": float(np.mean(row_mask)),
                "component_count": len(found),
                "expected_count": 8,
                "candidate_count": candidate_count,
                "selected_cut_count": len(partition.cut_positions),
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
            }
        )
    sheet = _handwritten_sheet()
    sheet_result, sheet_samples = _measure(
        partial(detect_spaced_regions, sheet, mode="dark"),
        repeat=repeat,
        warmup=warmup,
    )
    row_diagnostics = sheet_result.diagnostics.row_segmentation
    anchor_only = all(
        row.segmentation_strategy == "anchors" and not row.attempted_counts
        for row in row_diagnostics
    )
    workloads.append(
        {
            "name": "handwritten_sheet_7row",
            "shape": [sheet.height, sheet.width],
            "row_anchor_counts": [row.anchor_count for row in row_diagnostics],
            "body_anchors": sum(row.anchor_count or 0 for row in row_diagnostics),
            "anchors_requiring_repair": sum(
                len(row.split_anchor_indices) + len(row.merged_anchor_groups)
                for row in row_diagnostics
            ),
            "partition_states_visited": 0 if anchor_only else None,
            "truncated_local_searches": sum(len(row.truncated_counts) for row in row_diagnostics),
            "returned_regions": len(sheet_result.regions),
            "samples_seconds": sheet_samples,
            "median_seconds": statistics.median(sheet_samples),
        }
    )

    for name, image in _foreground_cases():
        operations = (
            ("mask_only", partial(foreground_mask, image, mode="auto")),
            ("evidence", partial(foreground_evidence, image, mode="auto")),
        )
        for api, operation in operations:
            result, samples = _measure(operation, repeat=repeat, warmup=warmup)
            decision_field = getattr(result, "decision_field", None)
            if api == "mask_only" and decision_field is not None:
                raise RuntimeError("foreground_mask unexpectedly returned a decision field")
            workloads.append(
                {
                    "name": f"{name}_{api}",
                    "shape": list(image.shape),
                    "pixels": int(image.size),
                    "input_bytes": int(image.nbytes),
                    "mode": "auto",
                    "foreground_fraction": float(result.mask.mean()),
                    "mask_bytes": int(result.mask.nbytes),
                    "decision_field_allocated": decision_field is not None,
                    "decision_field_bytes": (
                        0 if decision_field is None else int(decision_field.nbytes)
                    ),
                    "samples_seconds": samples,
                    "median_seconds": statistics.median(samples),
                }
            )

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeat": repeat,
        "warmup": warmup,
        "workloads": workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="measured runs per workload")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warmup runs per workload")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup must be nonnegative")
    print(
        json.dumps(run_benchmarks(repeat=args.repeat, warmup=args.warmup), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
