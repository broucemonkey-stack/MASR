"""General-purpose string and filename utilities with no I/O dependencies."""

from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    """Convert an arbitrary string into a URL-safe slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:48]


def sanitize_filename(filename: str, default: str = "file") -> str:
    """Clean a filename by removing dangerous characters and Windows reserved names."""
    name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._ ")
    if not name or name in {".", ".."}:
        name = default
    if Path(name).stem.upper() in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    return name[:160]


def unique_filename(directory: Path, filename: str) -> str:
    """Return a filename that does not already exist in *directory*.

    Appends ``_2``, ``_3``, … suffixes until a free name is found.
    """
    candidate = filename
    stem = Path(filename).stem or "file"
    suffix = Path(filename).suffix
    index = 2
    while (directory / candidate).exists():
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}
