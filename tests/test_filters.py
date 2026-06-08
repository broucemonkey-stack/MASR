from __future__ import annotations

from masr.filters import (
    ExperimentFilters,
    collect_dynamic_keys,
    collect_tags,
    filter_experiments,
    metric_range,
)
from masr.config_parser import extract_params_from_config_text, pick_default_param_keys
from masr.models import Experiment
from masr.parsing import format_key_value_lines, parse_key_value_lines, parse_tags


def test_filter_experiments_by_basic_fields_tags_params_and_metrics():
    experiments = [
        Experiment(
            id="a",
            name="baseline dice",
            tags=["baseline"],
            dataset="ISIC",
            model="UNet",
            strategy="no prompt",
            params={"loss": "dice", "learning_rate": 1e-4},
            metrics={"Dice": 0.86, "IoU": 0.75},
        ),
        Experiment(
            id="b",
            name="sam prompt",
            tags=["prompt"],
            dataset="Kvasir",
            model="SAM",
            strategy="box prompt",
            params={"loss": "ce", "learning_rate": 5e-5},
            metrics={"Dice": 0.91, "IoU": 0.84},
        ),
    ]

    filters = ExperimentFilters(
        search="dice",
        datasets={"ISIC"},
        models={"UNet"},
        tags={"baseline"},
        param_key="loss",
        param_value="dic",
        metric_key="Dice",
        metric_min=0.8,
        metric_max=0.9,
    )

    assert [experiment.id for experiment in filter_experiments(experiments, filters)] == ["a"]


def test_collectors_metric_range_and_parsing():
    experiments = [
        Experiment(
            id="a",
            name="A",
            tags=["x", "y"],
            params={"lr": 1e-4},
            metrics={"Dice": 0.86},
        ),
        Experiment(
            id="b",
            name="B",
            tags=["y"],
            params={"loss": "dice"},
            metrics={"Dice": "0.91", "Comment": "best"},
        ),
    ]

    assert collect_tags(experiments) == ["x", "y"]
    assert collect_dynamic_keys(experiments, "params") == ["loss", "lr"]
    assert metric_range(experiments, "Dice") == (0.86, 0.91)
    assert metric_range(experiments, "Comment") is None

    assert parse_tags("baseline, no_prompt，dice") == ["baseline", "no_prompt", "dice"]
    assert parse_key_value_lines("lr = 1e-4\nseed: 42\nuse_amp = true\nnote = keep") == {
        "lr": 0.0001,
        "seed": 42,
        "use_amp": True,
        "note": "keep",
    }

    payload = {"classes": ["1", "3"], "loss": {"type": "CrossEntropyLoss"}, "lr": 0.001}
    assert parse_key_value_lines(format_key_value_lines(payload)) == payload


def test_extract_params_from_python_config_text():
    params = extract_params_from_config_text(
        """
dataset_type = 'CustomDataset'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet', depth=50),
    head=dict(num_classes=6, loss=dict(type='CrossEntropyLoss')),
)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001))
train_cfg = dict(max_epochs=100, val_interval=1)
train_dataloader = dict(batch_size=64, dataset=dict(classes=['1', '3']))
work_dir = 'work_dirs/run'
"""
    )

    assert params["dataset_type"] == "CustomDataset"
    assert params["model.type"] == "ImageClassifier"
    assert params["model.backbone.depth"] == 50
    assert params["model.head.loss.type"] == "CrossEntropyLoss"
    assert params["optim_wrapper.optimizer.lr"] == 0.001
    assert params["train_dataloader.dataset.classes"] == ["1", "3"]
    assert pick_default_param_keys(sorted(params))[:3] == [
        "dataset_type",
        "model.type",
        "model.backbone.type",
    ]
