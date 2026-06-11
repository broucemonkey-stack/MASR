"""Tests for image thumbnail utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from masr.image_utils import generate_thumbnail, thumbnail_path


def test_generate_thumbnail_creates_thumbnail():
    """Thumbnail is created with correct dimensions."""
    img_dir = Path("test_artifacts") / "thumb_test"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a test image
    img = Image.new("RGB", (800, 600), color="red")
    img_path = img_dir / "test.png"
    img.save(img_path)

    thumb = generate_thumbnail(img_path, max_width=400)
    assert thumb is not None
    assert thumb.exists()
    assert thumb.parent.name == "thumbnails"

    thumb_img = Image.open(thumb)
    assert thumb_img.width <= 400


def test_generate_thumbnail_small_image_passthrough():
    """Small images (width <= max_width) are saved as-is but without resize."""
    img_dir = Path("test_artifacts") / "thumb_test_small"
    img_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (200, 150), color="blue")
    img_path = img_dir / "small.png"
    img.save(img_path)

    thumb = generate_thumbnail(img_path, max_width=400)
    assert thumb is not None
    assert thumb.exists()

    thumb_img = Image.open(thumb)
    # Small image kept at original size
    assert thumb_img.width == 200


def test_generate_thumbnail_missing_file_returns_none():
    """Non-existent source returns None."""
    result = generate_thumbnail(Path("nonexistent.png"))
    assert result is None


def test_thumbnail_path():
    """thumbnail_path returns the expected path inside thumbnails/."""
    images_dir = Path("/fake/images")
    expected = images_dir / "thumbnails" / "result.png"
    assert thumbnail_path(images_dir, "result.png") == expected
