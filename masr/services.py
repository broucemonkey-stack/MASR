"""Business-logic service layer for MASR.

Service classes add validation and orchestration on top of the
file-backed :class:`~masr.storage.AblationStore`.  The store remains
directly usable for low-level access; services are the recommended API
for UI and script consumers.
"""

from __future__ import annotations

from .models import Experiment, ImageRecord, Project
from .storage import AblationStore


class ExperimentService:
    """Business logic for experiment CRUD operations."""

    def __init__(self, store: AblationStore) -> None:
        self._store = store

    # -- read helpers --------------------------------------------------------

    def list_experiments(self, project_id: str) -> list[Experiment]:
        return self._store.list_experiments(project_id)

    def get_experiment(self, project_id: str, experiment_id: str) -> Experiment | None:
        return self._store.get_experiment(project_id, experiment_id)

    # -- write helpers -------------------------------------------------------

    def create_experiment(self, project_id: str, experiment: Experiment) -> Experiment:
        if not experiment.name.strip():
            raise ValueError("实验名称不能为空")
        return self._store.create_experiment(project_id, experiment)

    def save_experiment(self, project_id: str, experiment: Experiment) -> Experiment:
        if not experiment.name.strip():
            raise ValueError("实验名称不能为空")
        if not experiment.id:
            raise ValueError("实验ID不能为空 (实验尚未保存)")
        return self._store.save_experiment(project_id, experiment)

    def delete_experiment(self, project_id: str, experiment_id: str) -> None:
        self._store.delete_experiment(project_id, experiment_id)

    # -- config helpers ------------------------------------------------------

    def save_config_file(
        self,
        project_id: str,
        experiment_id: str,
        original_name: str,
        content: bytes,
    ) -> tuple[str, str]:
        return self._store.save_config_file(project_id, experiment_id, original_name, content)

    def read_config_text(self, project_id: str, experiment: Experiment) -> str | None:
        return self._store.read_config_text(project_id, experiment)

    # -- image helpers -------------------------------------------------------

    def save_image_file(
        self,
        project_id: str,
        experiment_id: str,
        original_name: str,
        content: bytes,
        label: str = "result",
        note: str = "",
    ) -> ImageRecord:
        return self._store.save_image_file(
            project_id, experiment_id, original_name, content,
            label=label, note=note,
        )

    def image_path(self, project_id: str, experiment_id: str, image: ImageRecord):
        return self._store.image_path(project_id, experiment_id, image)

    def thumbnail_path(self, project_id: str, experiment_id: str, image: ImageRecord):
        return self._store.thumbnail_path(project_id, experiment_id, image)


class ProjectService:
    """Business logic for project CRUD operations."""

    def __init__(self, store: AblationStore) -> None:
        self._store = store

    def list_projects(self) -> list[Project]:
        return self._store.list_projects()

    def get_project(self, project_id: str) -> Project | None:
        return self._store.get_project(project_id)

    def create_project(self, name: str, description: str = "") -> Project:
        if not name.strip():
            raise ValueError("项目名称不能为空")
        return self._store.create_project(name, description)

    def update_project(self, project: Project) -> Project:
        if not project.name.strip():
            raise ValueError("项目名称不能为空")
        return self._store.update_project(project)

    def delete_project(self, project_id: str) -> None:
        self._store.delete_project(project_id)
