"""Image thumbnail generation for MASR.

Generates lightweight thumbnails for uploaded result images so the
comparison grid loads faster.  Failures are non-fatal — callers should
fall back to the original image.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

THUMBNAIL_MAX_WIDTH = 300


def generate_thumbnail(source_path: Path, max_width: int = THUMBNAIL_MAX_WIDTH) -> Path | None:
    """Create a thumbnail for *source_path* and return its path.

    The thumbnail is written to a ``thumbnails/`` subdirectory alongside
    the source file.  The image is always scaled so that its width does
    not exceed *max_width* (aspect ratio preserved).  Returns ``None`` if
    generation fails for any reason (missing file, corrupt image, etc.).
    """
    thumbnail_dir = source_path.parent / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumbnail_dir / source_path.name

    try:
        img = Image.open(source_path)
        img.thumbnail((max_width, max_width * 10), Image.Resampling.LANCZOS)
        img.save(thumb_path, optimize=True)
        return thumb_path
    except Exception:
        return None


def thumbnail_path(images_dir: Path, filename: str) -> Path:
    """Return the expected thumbnail path for *filename* in *images_dir*."""
    return images_dir / "thumbnails" / filename
