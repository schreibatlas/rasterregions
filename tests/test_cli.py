from __future__ import annotations

import json

from PIL import Image, ImageDraw

from rasterregions.cli import main


def _two_row_page() -> Image.Image:
    image = Image.new("RGB", (220, 130), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 85):
        for x in (25, 155):
            draw.rectangle((x, y, x + 24, y + 24), fill="black")
    return image


def test_cli_page_scope_json_and_artifact_switches(tmp_path, capsys) -> None:
    image_path = tmp_path / "page.png"
    _two_row_page().save(image_path)
    output = tmp_path / "result"

    status = main(
        [
            "detect",
            str(image_path),
            "--output",
            str(output),
            "--expected-count",
            "4",
            "--expected-count-scope",
            "page",
            "--mode",
            "dark",
            "--json",
            "--no-crops",
            "--no-overlay",
            "--no-mask",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["status"] == "ok"
    assert payload["parameters"]["expected_count_scope"] == "page"
    assert payload["detected_count"] == 4
    assert (output / "regions.json").is_file()
    assert not (output / "crops").exists()
    assert not (output / "regions.png").exists()
    assert not (output / "mask.png").exists()


def test_cli_failed_detection_keeps_nonzero_exit_status(tmp_path, capsys) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (30, 30), "white").save(image_path)

    status = main(
        [
            "detect",
            str(image_path),
            "--output",
            str(tmp_path / "result"),
            "--expected-count",
            "1",
            "--mode",
            "dark",
        ]
    )

    assert status == 2
    assert "status: failed" in capsys.readouterr().out
