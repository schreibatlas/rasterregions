from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .detect import detect_regions
from .io import detection_payload, write_detection


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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
