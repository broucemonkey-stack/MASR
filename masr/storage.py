from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Experiment, ImageRecord, Project


class AblationStore:
    """File-backed storage for projects, experiment manifests, configs, and images."""

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        self.projects_root = self.root / "projects"

    def ensure(self) -> None:
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[Project]:
        self.ensure()
        projects: list[Project] = []
        for project_file in sorted(self.projects_root.glob("*/project.json")):
            try:
                projects.append(Project.from_dict(_read_json(project_file)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def get_project(self, project_id: str) -> Project | None:
        project_file = self.project_dir(project_id) / "project.json"
        if not project_file.exists():
            return None
        return Project.from_dict(_read_json(project_file))

    def create_project(self, name: str, description: str = "") -> Project:
        self.ensure()
        now = now_iso()
        project = Project(
            id=make_id(name),
            name=name.strip(),
            description=description.strip(),
            created_at=now,
            updated_at=now,
        )
        self.project_dir(project.id).mkdir(parents=True, exist_ok=False)
        self.experiments_dir(project.id).mkdir(parents=True, exist_ok=True)
        _write_json(self.project_dir(project.id) / "project.json", project.to_dict())
        return project

    def update_project(self, project: Project) -> Project:
        existing = self.get_project(project.id)
        if existing is None:
            raise FileNotFoundError(f"Project not found: {project.id}")
        project.created_at = existing.created_at
        project.updated_at = now_iso()
        _write_json(self.project_dir(project.id) / "project.json", project.to_dict())
        return project

    def delete_project(self, project_id: str) -> None:
        target = self.project_dir(project_id)
        if target.exists():
            _assert_inside(target, self.projects_root)
            shutil.rmtree(target)

    def list_experiments(self, project_id: str) -> list[Experiment]:
        project_experiments = self.experiments_dir(project_id)
        if not project_experiments.exists():
            return []
        experiments: list[Experiment] = []
        for manifest in sorted(project_experiments.glob("*/manifest.json")):
            try:
                experiments.append(Experiment.from_dict(_read_json(manifest)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return sorted(experiments, key=lambda experiment: experiment.updated_at, reverse=True)

    def get_experiment(self, project_id: str, experiment_id: str) -> Experiment | None:
        manifest = self.experiment_dir(project_id, experiment_id) / "manifest.json"
        if not manifest.exists():
            return None
        return Experiment.from_dict(_read_json(manifest))

    def create_experiment(self, project_id: str, experiment: Experiment) -> Experiment:
        if self.get_project(project_id) is None:
            raise FileNotFoundError(f"Project not found: {project_id}")
        now = now_iso()
        experiment.id = experiment.id or make_id(experiment.name)
        experiment.created_at = now
        experiment.updated_at = now
        experiment.images = experiment.images or []

        exp_dir = self.experiment_dir(project_id, experiment.id)
        exp_dir.mkdir(parents=True, exist_ok=False)
        (exp_dir / "images").mkdir(parents=True, exist_ok=True)
        self.save_experiment(project_id, experiment)
        self.touch_project(project_id)
        return experiment

    def save_experiment(self, project_id: str, experiment: Experiment) -> Experiment:
        if not self.experiment_dir(project_id, experiment.id).exists():
            raise FileNotFoundError(f"Experiment not found: {experiment.id}")
        existing = self.get_experiment(project_id, experiment.id)
        if existing is not None:
            experiment.created_at = existing.created_at
        experiment.updated_at = now_iso()
        _write_json(
            self.experiment_dir(project_id, experiment.id) / "manifest.json",
            experiment.to_dict(),
        )
        self.touch_project(project_id)
        return experiment

    def delete_experiment(self, project_id: str, experiment_id: str) -> None:
        target = self.experiment_dir(project_id, experiment_id)
        if target.exists():
            _assert_inside(target, self.experiments_dir(project_id))
            shutil.rmtree(target)
            self.touch_project(project_id)

    def save_config_file(
        self,
        project_id: str,
        experiment_id: str,
        original_name: str,
        content: bytes,
    ) -> tuple[str, str]:
        exp_dir = self.experiment_dir(project_id, experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        filename = "config.py"
        (exp_dir / filename).write_bytes(content)
        return filename, sanitize_filename(original_name, default="config.py")

    def read_config_text(self, project_id: str, experiment: Experiment) -> str | None:
        if not experiment.config_file:
            return None
        path = self.experiment_dir(project_id, experiment.id) / experiment.config_file
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def save_image_file(
        self,
        project_id: str,
        experiment_id: str,
        original_name: str,
        content: bytes,
        label: str = "result",
        note: str = "",
    ) -> ImageRecord:
        images_dir = self.experiment_dir(project_id, experiment_id) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        filename = unique_filename(images_dir, sanitize_filename(original_name, default="image.png"))
        (images_dir / filename).write_bytes(content)
        return ImageRecord(
            filename=filename,
            label=label.strip() or "result",
            note=note.strip(),
            uploaded_at=now_iso(),
        )

    def image_path(self, project_id: str, experiment_id: str, image: ImageRecord) -> Path:
        return self.experiment_dir(project_id, experiment_id) / "images" / image.filename

    def project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def experiments_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "experiments"

    def experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        return self.experiments_dir(project_id) / experiment_id

    def touch_project(self, project_id: str) -> None:
        project = self.get_project(project_id)
        if project is not None:
            project.updated_at = now_iso()
            _write_json(self.project_dir(project_id) / "project.json", project.to_dict())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_id(name: str) -> str:
    stem = slugify(name) or "item"
    return f"{stem}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:48]


def sanitize_filename(filename: str, default: str = "file") -> str:
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
    candidate = filename
    stem = Path(filename).stem or "file"
    suffix = Path(filename).suffix
    index = 2
    while (directory / candidate).exists():
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _assert_inside(target: Path, root: Path) -> None:
    target_resolved = target.resolve()
    root_resolved = root.resolve()
    if root_resolved not in target_resolved.parents and target_resolved != root_resolved:
        raise ValueError(f"Refusing to remove path outside storage root: {target}")


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
