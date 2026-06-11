"""Tests for data model validation and depth protection."""

from __future__ import annotations

import pytest

from masr.models import Experiment, ImageRecord, Project


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


def test_project_empty_name_raises():
    with pytest.raises(ValueError, match="Project name must not be empty"):
        Project(id="x", name="")


def test_project_whitespace_name_raises():
    with pytest.raises(ValueError, match="Project name must not be empty"):
        Project(id="x", name="   ")


def test_project_valid_name_ok():
    p = Project(id="x", name="SAM Ablation")
    assert p.name == "SAM Ablation"


def test_experiment_empty_name_raises():
    with pytest.raises(ValueError, match="Experiment name must not be empty"):
        Experiment(id="e", name="")


def test_experiment_whitespace_name_raises():
    with pytest.raises(ValueError, match="Experiment name must not be empty"):
        Experiment(id="e", name="\t\n")


def test_experiment_valid_name_ok():
    e = Experiment(id="e", name="baseline")
    assert e.name == "baseline"


def test_image_record_empty_filename_raises():
    with pytest.raises(ValueError, match="Image filename must not be empty"):
        ImageRecord(filename="")


def test_image_record_valid_filename_ok():
    img = ImageRecord(filename="result.png")
    assert img.filename == "result.png"


# ---------------------------------------------------------------------------
# max_depth protection
# ---------------------------------------------------------------------------


def test_experiment_from_dict_max_depth_exceeded():
    """At depth 0, the first ImageRecord.from_dict call hits depth 1 > 0."""
    nested = {
        "id": "x",
        "name": "test",
        "images": [{"filename": "a.png", "label": "result", "note": "", "uploaded_at": ""}],
    }
    with pytest.raises(ValueError, match="Maximum nesting depth"):
        Experiment.from_dict(nested, max_depth=0)


def test_experiment_from_dict_default_depth_ok():
    nested = {
        "id": "x",
        "name": "test",
        "images": [{"filename": "a.png", "label": "result", "note": "", "uploaded_at": ""}],
    }
    exp = Experiment.from_dict(nested)
    assert exp.name == "test"
    assert len(exp.images) == 1


def test_project_from_dict_max_depth_exceeded():
    with pytest.raises(ValueError, match="Maximum nesting depth"):
        Project.from_dict({"id": "x", "name": "test"}, max_depth=-1)


def test_image_from_dict_max_depth_exceeded():
    with pytest.raises(ValueError, match="Maximum nesting depth"):
        ImageRecord.from_dict({"filename": "x.png"}, max_depth=-1)
