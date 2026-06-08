from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from masr.models import Experiment
from masr.storage import AblationStore, sanitize_filename, unique_filename


def test_project_and_experiment_lifecycle(monkeypatch):
    store = AblationStore(_fresh_root())

    project = store.create_project("SAM Ablation", "baseline variants")
    assert project.name == "SAM Ablation"
    assert store.get_project(project.id).description == "baseline variants"

    experiment = Experiment(
        id="",
        name="No prompt baseline",
        description="first run",
        tags=["baseline", "no_prompt"],
        dataset="ISIC",
        model="UNet",
        strategy="no prompt",
        seed="42",
        params={"learning_rate": 1e-4, "loss": "dice"},
        metrics={"Dice": 0.871, "IoU": 0.792},
    )
    experiment = store.create_experiment(project.id, experiment)

    config_file, original_name = store.save_config_file(
        project.id,
        experiment.id,
        "../config run.py",
        b"learning_rate = 1e-4\n",
    )
    image = store.save_image_file(
        project.id,
        experiment.id,
        "result:mask?.png",
        b"not really an image",
        label="mask",
        note="prediction",
    )
    experiment.config_file = config_file
    experiment.config_original_name = original_name
    experiment.images.append(image)
    store.save_experiment(project.id, experiment)

    loaded = store.get_experiment(project.id, experiment.id)
    assert loaded is not None
    assert loaded.params["loss"] == "dice"
    assert loaded.metrics["Dice"] == 0.871
    assert loaded.images[0].filename == "result_mask_.png"
    assert store.read_config_text(project.id, loaded) == "learning_rate = 1e-4\n"
    assert store.image_path(project.id, loaded.id, loaded.images[0]).read_bytes() == b"not really an image"

    assert [item.id for item in store.list_experiments(project.id)] == [experiment.id]

    loaded.name = "Edited baseline"
    loaded.dataset = "Edited dataset"
    loaded.params["learning_rate"] = 5e-5
    loaded.metrics["Dice"] = 0.9
    store.save_experiment(project.id, loaded)

    edited = store.get_experiment(project.id, experiment.id)
    assert edited.name == "Edited baseline"
    assert edited.dataset == "Edited dataset"
    assert edited.params["learning_rate"] == 5e-5
    assert edited.metrics["Dice"] == 0.9

    removed_paths = []

    def fake_rmtree(path):
        removed_paths.append(Path(path))

    monkeypatch.setattr("masr.storage.shutil.rmtree", fake_rmtree)
    store.delete_experiment(project.id, experiment.id)
    store.delete_project(project.id)

    assert removed_paths == [
        store.experiment_dir(project.id, experiment.id),
        store.project_dir(project.id),
    ]


def test_list_experiments_skips_broken_manifest():
    store = AblationStore(_fresh_root())
    project = store.create_project("Project")
    good = store.create_experiment(project.id, Experiment(id="", name="Good"))

    broken_dir = store.experiments_dir(project.id) / "broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "manifest.json").write_text("{broken", encoding="utf-8")

    assert [experiment.id for experiment in store.list_experiments(project.id)] == [good.id]


def test_sanitize_filename_and_unique_filename():
    assert sanitize_filename("../../bad: name?.png") == "bad_name_.png"
    assert sanitize_filename("CON.py") == "_CON.py"
    assert sanitize_filename("") == "file"

    directory = _fresh_root() / "images"
    directory.mkdir()
    (directory / "plot.png").write_bytes(b"existing")

    assert unique_filename(directory, "plot.png") == "plot_2.png"


def _fresh_root() -> Path:
    root = Path("test_artifacts") / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root
