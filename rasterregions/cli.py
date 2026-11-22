from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .detect import detect_regions, detect_spaced_region_hypotheses
from .io import (
    detection_payload,
    spaced_hypotheses_payload,
    write_detection,
    write_spaced_hypotheses,
)


def _parse_row_counts(value: str) -> tuple[int | None, ...]:
    counts: list[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if token in {"auto", "none"}:
            counts.append(None)
            continue
        try:
            count = int(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "row counts must be positive integers or 'auto'"
            ) from error
        if count < 1:
            raise argparse.ArgumentTypeError("row counts must be positive integers or 'auto'")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one row count is required")
    return tuple(counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rasterregions")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect", help="detect anonymous regions in an image")
    detect.add_argument("image", type=Path)
    detect.add_argument("--output", "-o", type=Path, required=True)
    detect.add_argument("--expected-count", type=int, default=None)
    detect.add_argument("--expected-count-scope", choices=("row", "page"), default="row")
    detect.add_argument(
        "--mode",
        choices=("auto", "grayscale", "dark", "color", "hybrid", "adaptive"),
        default="auto",
    )
    detect.add_argument("--padding", type=int, default=None)
    detect.add_argument("--min-confidence", type=float, default=0.3)
    detect.add_argument("--json", action="store_true", dest="json_output")
    detect.add_argument("--no-crops", action="store_false", dest="include_crops", default=True)
    detect.add_argument("--no-overlay", action="store_false", dest="include_overlay", default=True)
    detect.add_argument("--no-mask", action="store_false", dest="include_mask", default=True)
    spaced = sub.add_parser("spaced", help="segment sparse regions and return top-K hypotheses")
    spaced.add_argument("image", type=Path)
    spaced.add_argument("--output", "-o", type=Path, required=True)
    spaced.add_argument("--expected-count", type=int, default=None)
    spaced.add_argument("--row-counts", type=_parse_row_counts, default=None)
    spaced.add_argument(
        "--mode",
        choices=("auto", "grayscale", "dark", "color", "hybrid", "adaptive"),
        default="auto",
    )
    spaced.add_argument("--min-confidence", type=float, default=0.3)
    spaced.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        result = detect_regions(
            args.image,
            expected_count=args.expected_count,
            expected_count_scope=args.expected_count_scope,
            mode=args.mode,
            padding=args.padding,
            min_confidence=args.min_confidence,
        )
        write_detection(
            result,
            args.output,
            include_crops=args.include_crops,
            include_overlay=args.include_overlay,
            include_mask=args.include_mask,
        )
        if args.json_output:
            print(json.dumps(detection_payload(result), ensure_ascii=False))
        else:
            print(f"status: {result.status}")
            print(f"regions: {len(result.regions)}")
            print(f"manifest: {args.output / 'regions.json'}")
        return 0 if result.status != "failed" else 2
    if args.command == "spaced":
        hypotheses = detect_spaced_region_hypotheses(
            args.image,
            expected_count=args.expected_count,
            row_counts=args.row_counts,
            mode=args.mode,
            min_confidence=args.min_confidence,
        )
        manifest = write_spaced_hypotheses(hypotheses, args.output)
        if args.json_output:
            print(json.dumps(spaced_hypotheses_payload(hypotheses), ensure_ascii=False))
        else:
            print(f"status: {hypotheses.selected.status}")
            print(f"regions: {len(hypotheses.selected.regions)}")
            print(f"page hypotheses: {len(hypotheses.page_hypotheses)}")
            print(f"search truncated: {hypotheses.search_truncated}")
            print(f"alternatives omitted: {hypotheses.alternatives_omitted}")
            if hypotheses.search_limit:
                print(f"search limit: {hypotheses.search_limit}")
            if hypotheses.failure_reason:
                print(f"failure: {hypotheses.failure_reason}")
            print(f"manifest: {manifest}")
        return 0 if hypotheses.selected.status != "failed" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
