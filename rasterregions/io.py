from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import (
    DetectionResult,
    PagePartitionHypothesis,
    RegionHypothesis,
    RowPartitionHypothesis,
    SpacedSegmentationHypotheses,
)


def detection_payload(result: DetectionResult) -> dict:
    """Return the schema-versioned JSON-compatible representation of ``result``.

    Component indexes in the payload use the ``raw_mask`` namespace.
    """
    from . import __version__

    diagnostics = result.diagnostics
    decisions = diagnostics.component_decisions if diagnostics else result.component_decisions
    boundaries = diagnostics.partition_boundaries if diagnostics else ()
    assert result.unassigned_mask is not None
    width, height = diagnostics.image_size if diagnostics else result.image.size
    payload = {
        "rasterregions_version": __version__,
        "schema_version": 1,
        "component_index_namespace": "raw_mask",
        "source": result.source_path.name if result.source_path else None,
        "status": result.status,
        "expected_count": result.expected_count,
        "detected_count": len(result.regions),
        "image": {"width": width, "height": height},
        "parameters": {
            "mode": diagnostics.mode if diagnostics else "auto",
            "padding": diagnostics.padding if diagnostics else None,
            "min_confidence": diagnostics.min_confidence if diagnostics else 0.3,
            "expected_count": result.expected_count,
            "expected_count_scope": result.expected_count_scope,
            "roi": list(diagnostics.roi.as_tuple()) if diagnostics and diagnostics.roi else None,
        },
        "mask": {
            "method": result.mask_result.method,
            "threshold": result.mask_result.threshold,
            "confidence": round(result.mask_result.confidence, 4),
            "foreground_fraction": round(float(result.mask_result.mask.mean()), 6),
            "warnings": list(result.mask_result.warnings),
        },
        "cleanup": [
            {
                "component_index": decision.component_index,
                "bbox": list(decision.bbox.as_tuple()),
                "area": decision.area,
                "role": decision.role,
                "action": decision.action,
                "reason": decision.reason,
            }
            for decision in decisions
        ],
        "rows": [
            {
                "row": index,
                "bbox": list(row.bbox.as_tuple()),
                "component_indices": list(row.component_indices),
            }
            for index, row in enumerate(result.rows)
        ],
        "partition_boundaries": [
            {
                "x": boundary.x,
                "projection": boundary.projection,
                "crossing_components": boundary.crossing_components,
                "confidence": round(boundary.confidence, 4),
            }
            for boundary in boundaries
        ],
        "warnings": list(result.warnings),
        "regions": [
            {
                "id": region.region_id,
                "row": region.row_index,
                "column": region.column_index,
                "reading_order": region.reading_order,
                "bbox": list(region.bbox.as_tuple()),
                "confidence": round(region.confidence, 4),
                "component_indices": list(region.component_indices),
                "primary_component_indices": list(region.primary_component_indices),
                "secondary_component_indices": list(region.secondary_component_indices),
                "ambiguous_component_indices": list(region.ambiguous_component_indices),
            }
            for region in result.regions
        ],
    }
    if diagnostics is not None and (
        diagnostics.row_count is not None
        or diagnostics.row_segmentation
        or diagnostics.secondary_mark_assignments
    ):
        payload["spacing"] = {
            "selected_region_count": diagnostics.selected_region_count,
            "row_count": diagnostics.row_count,
            "best_second_best_partition_margin": diagnostics.best_second_best_partition_margin,
            "unassigned_ink_fraction": round(diagnostics.unassigned_ink_fraction, 6),
            "unassigned_ink_pixels": int(result.unassigned_mask.sum()),
            "search_truncated": diagnostics.search_truncated,
            "search_limit": diagnostics.search_limit,
            "width_prior_deviation": diagnostics.width_prior_deviation,
            "body_counts_per_region": list(diagnostics.body_counts_per_region),
            "rows": [
                {
                    "row_index": row.row_index,
                    "bbox": list(row.bbox.as_tuple()),
                    "component_indices": list(row.component_indices),
                    "y_range": list(row.y_range),
                    "center_y": round(row.center_y, 4),
                    "baseline_y": round(row.baseline_y, 4),
                    "body_height": round(row.body_height, 4),
                    "width_prior_deviation": row.width_prior_deviation,
                    "body_counts_per_region": list(row.body_counts_per_region),
                    "anchor_count": row.anchor_count,
                    "segmentation_strategy": row.segmentation_strategy,
                    "attempted_counts": list(row.attempted_counts),
                    "truncated_counts": list(row.truncated_counts),
                    "split_anchor_indices": list(row.split_anchor_indices),
                    "merged_anchor_groups": [list(group) for group in row.merged_anchor_groups],
                    "inferred_count": row.inferred_count,
                    "requested_count": row.requested_count,
                    "selected_boundaries": list(row.selected_boundaries),
                    "confidence": round(row.confidence, 4),
                    "cost": round(row.cost, 4),
                    "alternative_cost_margin": (
                        round(row.alternative_cost_margin, 4)
                        if row.alternative_cost_margin is not None
                        else None
                    ),
                    "ambiguous_component_indices": list(row.ambiguous_component_indices),
                    "warnings": list(row.warnings),
                }
                for row in diagnostics.row_segmentation
            ],
            "candidate_boundaries": [
                {
                    "x": boundary.x,
                    "projection": boundary.projection,
                    "crossing_components": boundary.crossing_components,
                    "component_indices": list(boundary.component_indices),
                    "selected": boundary.selected,
                    "kind": boundary.kind,
                    "confidence": round(boundary.confidence, 4),
                    "valley_strength": boundary.valley_strength,
                }
                for boundary in diagnostics.partition_boundaries
            ],
            "secondary_mark_assignments": [
                {
                    "component_index": assignment.component_index,
                    "target_region_index": assignment.target_region_index,
                    "best_score": assignment.best_score,
                    "second_best_score": assignment.second_best_score,
                    "margin": assignment.margin,
                    "status": assignment.status,
                    "candidate_region_indices": list(assignment.candidate_region_indices),
                }
                for assignment in diagnostics.secondary_mark_assignments
            ],
        }
    return payload


def _mask_runs(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for y, row in enumerate(np.asarray(mask, dtype=bool)):
        xs = np.flatnonzero(row)
        if not len(xs):
            continue
        start = previous = int(xs[0])
        for value in xs[1:]:
            x = int(value)
            if x != previous + 1:
                runs.append([y, start, previous + 1])
                start = x
            previous = x
        runs.append([y, start, previous + 1])
    return runs


def _boundary_payload(boundary) -> dict:
    return {
        "x": boundary.x,
        "kind": boundary.kind,
        "selected": boundary.selected,
        "projection": round(boundary.projection, 6),
        "valley_strength": boundary.valley_strength,
        "crossing_components": boundary.crossing_components,
        "component_indices": list(boundary.component_indices),
        "crossing_cost": round(boundary.crossing_cost, 6),
        "confidence": round(boundary.confidence, 6),
    }


def _assignment_payload(assignment) -> dict:
    return {
        "component_index": assignment.component_index,
        "target_region_index": assignment.target_region_index,
        "best_score": assignment.best_score,
        "second_best_score": assignment.second_best_score,
        "margin": assignment.margin,
        "status": assignment.status,
        "candidate_region_indices": list(assignment.candidate_region_indices),
    }


def _region_hypothesis_payload(region: RegionHypothesis) -> dict:
    return {
        "bbox": list(region.bbox.as_tuple()),
        "width": region.bbox.width,
        "height": region.bbox.height,
        "area": int(region.mask.sum()),
        "mask_runs": _mask_runs(region.mask),
        "component_indices": list(region.component_indices),
        "primary_component_indices": list(region.primary_component_indices),
        "secondary_component_indices": list(region.secondary_component_indices),
        "ambiguous_component_indices": list(region.ambiguous_component_indices),
        "body_count": region.body_count,
        "frame": {
            "row_index": region.frame.row_index,
            "baseline_y": round(region.frame.baseline_y, 4),
            "body_height": round(region.frame.body_height, 4),
            "row_center_y": round(region.frame.row_center_y, 4),
        },
        "cost": round(region.cost, 6),
        "cut_crossing_cost": round(region.cut_crossing_cost, 6),
        "confidence": round(region.confidence, 6),
        "warnings": list(region.warnings),
    }


def _row_hypothesis_payload(row: RowPartitionHypothesis) -> dict:
    return {
        "row_index": row.row_index,
        "count": row.count,
        "boundaries": list(row.boundaries),
        "boundary_evidence": [_boundary_payload(boundary) for boundary in row.boundary_evidence],
        "cost": round(row.cost, 6),
        "cost_components": {
            "row_partition": round(row.cost, 6),
            "cut_crossing": round(row.cut_crossing_cost, 6),
            "unassigned_ink_fraction": round(row.unassigned_ink_fraction, 6),
        },
        "unassigned_ink_fraction": round(row.unassigned_ink_fraction, 6),
        "regions": [_region_hypothesis_payload(region) for region in row.regions],
        "secondary_mark_assignments": [
            _assignment_payload(assignment) for assignment in row.secondary_mark_assignments
        ],
    }


def _page_hypothesis_payload(
    page: PagePartitionHypothesis,
    cleaned_foreground_pixels: int,
) -> dict:
    unassigned = page.unassigned_mask
    assert unassigned is not None
    row_cost = sum(row.cost for row in page.row_hypotheses)
    return {
        "count": page.count,
        "allocation": [row.count for row in page.row_hypotheses],
        "cost": round(page.cost, 6),
        "cost_components": {
            "row_partitions": round(row_cost, 6),
            "secondary_mark_and_ownership": round(
                max(0.0, page.cost - row_cost - page.unassigned_ink_fraction), 6
            ),
            "unassigned_ink_fraction": round(page.unassigned_ink_fraction, 6),
        },
        "unassigned_ink_fraction": round(page.unassigned_ink_fraction, 6),
        "unassigned_ink_pixels": int(unassigned.sum()),
        "unassigned_mask_runs": _mask_runs(unassigned),
        "cleaned_foreground_pixels": cleaned_foreground_pixels,
        "rows": [_row_hypothesis_payload(row) for row in page.row_hypotheses],
    }


def spaced_hypotheses_payload(
    result: SpacedSegmentationHypotheses,
) -> dict:
    """Return JSON-compatible selected, alternative, and search diagnostics."""
    selected = result.selected
    diagnostics = selected.diagnostics
    payload = {
        "schema_version": 1,
        "segmentation": "spaced",
        "selected": detection_payload(selected),
        "row_hypotheses": [
            [_row_hypothesis_payload(row) for row in options] for options in result.row_hypotheses
        ],
        "page_hypotheses": [
            _page_hypothesis_payload(page, int(selected.clean_mask.sum()))
            for page in result.page_hypotheses
        ],
        "search": {
            "truncated": result.truncated,
            "limit": diagnostics.search_limit if diagnostics else None,
            "failure_reason": result.failure_reason,
        },
    }
    return payload


def _atomic_save_image(image: Image.Image, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        image.save(temporary_path, format="PNG")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_text(text: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _remove_stale_crops(crops: Path, *, remove_directory: bool) -> None:
    if crops.is_dir():
        for path in crops.glob("region-*.png"):
            if path.is_file():
                path.unlink()
        if remove_directory:
            with contextlib.suppress(OSError):
                crops.rmdir()


def write_detection(
    result: DetectionResult,
    output_dir: str | Path,
    *,
    include_crops: bool = True,
    include_overlay: bool = True,
    include_mask: bool = True,
) -> Path:
    """Write selected image artifacts and publish ``regions.json`` last.

    Reruns remove stale managed crops and disabled mask/overlay artifacts while
    leaving unrelated files in the output directory untouched."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    crops = output / "crops"
    _remove_stale_crops(crops, remove_directory=not include_crops)
    if include_crops:
        crops.mkdir(parents=True, exist_ok=True)
        for region in result.regions:
            _atomic_save_image(region.image, crops / f"{region.region_id}.png")

    mask_path = output / "mask.png"
    if include_mask:
        mask = Image.fromarray(np.asarray(result.clean_mask, dtype=np.uint8) * 255)
        _atomic_save_image(mask, mask_path)
    else:
        mask_path.unlink(missing_ok=True)

    overlay_path = output / "regions.png"
    if include_overlay:
        overlay = result.image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for region in result.regions:
            x0, y0, x1, y1 = region.bbox.as_tuple()
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(230, 45, 35), width=2)
            draw.text((x0, max(0, y0 - 12)), region.region_id, fill=(25, 75, 220))
        _atomic_save_image(overlay, overlay_path)
    else:
        overlay_path.unlink(missing_ok=True)

    manifest = output / "regions.json"
    payload = json.dumps(detection_payload(result), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(payload, manifest)
    return manifest


def _save_page_ownership_overlay(
    image: Image.Image,
    page: PagePartitionHypothesis,
    destination: Path,
) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    colors = ((230, 70, 70), (40, 160, 80), (40, 100, 220), (210, 130, 20))
    for index, region in enumerate(page.regions):
        color = colors[index % len(colors)]
        pixels = [
            (region.bbox.x0 + int(x), region.bbox.y0 + int(y))
            for y, x in zip(*np.nonzero(region.mask), strict=True)
        ]
        draw.point(pixels, fill=color)
        draw.rectangle(
            (
                region.bbox.x0,
                region.bbox.y0,
                region.bbox.x1 - 1,
                region.bbox.y1 - 1,
            ),
            outline=color,
            width=1,
        )
    if page.unassigned_mask is not None:
        ys, xs = np.nonzero(page.unassigned_mask)
        draw.point(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(245, 150, 20))
    _atomic_save_image(overlay, destination)


def write_spaced_hypotheses(
    result: SpacedSegmentationHypotheses,
    output_dir: str | Path,
) -> Path:
    """Write hypothesis JSON and selected/alternative ownership overlays."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = result.selected
    clean = Image.fromarray(np.asarray(selected.clean_mask, dtype=np.uint8) * 255)
    _atomic_save_image(clean, output / "cleaned_mask.png")

    row_overlay = selected.image.convert("RGB")
    row_draw = ImageDraw.Draw(row_overlay)
    for row in selected.rows:
        x0, y0, x1, y1 = row.bbox.as_tuple()
        row_draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(30, 110, 220), width=1)
    _atomic_save_image(row_overlay, output / "rows.png")

    boundary_overlay = selected.image.convert("RGB")
    boundary_draw = ImageDraw.Draw(boundary_overlay)
    diagnostics = selected.diagnostics
    for boundary in diagnostics.partition_boundaries if diagnostics else ():
        row_diagnostic = (
            next(
                (
                    item
                    for item in diagnostics.row_segmentation
                    if boundary.x in item.selected_boundaries
                ),
                None,
            )
            if diagnostics
            else None
        )
        y0, y1 = row_diagnostic.y_range if row_diagnostic else (0, selected.image.height)
        color = (230, 40, 40) if boundary.selected else (130, 130, 130)
        boundary_draw.line((boundary.x, y0, boundary.x, y1), fill=color, width=1)
    if diagnostics is not None:
        anchor_row_indices = {
            row.row_index
            for row in diagnostics.row_segmentation
            if row.segmentation_strategy in {"anchors", "hybrid"}
        }
        for region in selected.regions:
            if region.row_index in anchor_row_indices:
                boundary_draw.rectangle(
                    (
                        region.bbox.x0,
                        region.bbox.y0,
                        region.bbox.x1 - 1,
                        region.bbox.y1 - 1,
                    ),
                    outline=(35, 110, 220),
                    width=1,
                )
    _atomic_save_image(boundary_overlay, output / "candidate_cuts.png")

    if result.page_hypotheses:
        _save_page_ownership_overlay(
            selected.image,
            result.page_hypotheses[0],
            output / "selected_partition.png",
        )
        for index, page in enumerate(result.page_hypotheses[1:3], start=2):
            _save_page_ownership_overlay(
                selected.image,
                page,
                output / f"alternative_partition_{index}.png",
            )
        unassigned = result.page_hypotheses[0].unassigned_mask
    else:
        unassigned = selected.unassigned_mask
    if unassigned is not None:
        unassigned_image = Image.fromarray(np.asarray(unassigned, dtype=np.uint8) * 255)
        _atomic_save_image(unassigned_image, output / "unassigned_ink.png")

    manifest = output / "hypotheses.json"
    payload = json.dumps(spaced_hypotheses_payload(result), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(payload, manifest)
    return manifest
