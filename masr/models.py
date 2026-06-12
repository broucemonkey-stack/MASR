from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Project:
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Project name must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], max_depth: int = 10, _depth: int = 0) -> "Project":
        if _depth > max_depth:
            raise ValueError(f"Maximum nesting depth ({max_depth}) exceeded in Project.from_dict")
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ImageRecord:
    filename: str
    label: str = "result"
    note: str = ""
    uploaded_at: str = ""

    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise ValueError("Image filename must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], max_depth: int = 10, _depth: int = 0) -> "ImageRecord":
        if _depth > max_depth:
            raise ValueError(f"Maximum nesting depth ({max_depth}) exceeded in ImageRecord.from_dict")
        return cls(
            filename=str(payload.get("filename", "")),
            label=str(payload.get("label", "result") or "result"),
            note=str(payload.get("note", "")),
            uploaded_at=str(payload.get("uploaded_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "label": self.label,
            "note": self.note,
            "uploaded_at": self.uploaded_at,
        }


@dataclass(slots=True)
class Experiment:
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    dataset: str = ""
    model: str = ""
    strategy: str = ""
    seed: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    images: list[ImageRecord] = field(default_factory=list)
    config_file: str | None = None
    config_original_name: str | None = None
    log_file: str | None = None
    log_original_name: str | None = None
    test_log_file: str | None = None
    test_log_original_name: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any], max_depth: int = 10, _depth: int = 0) -> "Experiment":
        if _depth > max_depth:
            raise ValueError(f"Maximum nesting depth ({max_depth}) exceeded in Experiment.from_dict")
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            tags=_coerce_tags(payload.get("tags", [])),
            dataset=str(payload.get("dataset", "")),
            model=str(payload.get("model", "")),
            strategy=str(payload.get("strategy", "")),
            seed=str(payload.get("seed", "")),
            params=dict(payload.get("params") or {}),
            metrics=dict(payload.get("metrics") or {}),
            test_metrics=dict(payload.get("test_metrics") or {}),
            images=[ImageRecord.from_dict(item, max_depth=max_depth, _depth=_depth + 1) for item in payload.get("images", [])],
            config_file=payload.get("config_file"),
            config_original_name=payload.get("config_original_name"),
            log_file=payload.get("log_file"),
            log_original_name=payload.get("log_original_name"),
            test_log_file=payload.get("test_log_file"),
            test_log_original_name=payload.get("test_log_original_name"),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "dataset": self.dataset,
            "model": self.model,
            "strategy": self.strategy,
            "seed": self.seed,
            "params": self.params,
            "metrics": self.metrics,
            "test_metrics": self.test_metrics,
            "images": [image.to_dict() for image in self.images],
            "config_file": self.config_file,
            "config_original_name": self.config_original_name,
            "log_file": self.log_file,
            "log_original_name": self.log_original_name,
            "test_log_file": self.test_log_file,
            "test_log_original_name": self.test_log_original_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _coerce_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]
