from __future__ import annotations

from masr.filters import (
    ExperimentFilters,
    collect_dynamic_keys,
    collect_tags,
    filter_experiments,
    metric_range,
)
from masr.config_parser import (
    extract_params_from_config_text,
    extract_pipeline_summaries,
    pick_default_param_keys,
)
from masr.filters import to_float
from masr.models import Experiment
from masr.parsing import (
    format_key_value_lines,
    parse_key_value_lines,
    parse_metric_line,
    parse_metrics_text,
    parse_tags,
)


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
    # 新优先级：学习率 > 优化器 > 调度器 > pipeline > 通用
    top3 = pick_default_param_keys(sorted(params))[:3]
    assert "optim_wrapper.optimizer.lr" in top3
    assert "optim_wrapper.optimizer.type" in top3


# ---------------------------------------------------------------------------
# parse_metric_line / parse_metrics_text
# ---------------------------------------------------------------------------


def test_parse_metric_line_composite_format():
    """典型 MMEngine 评估日志行：验证集 + single-label 指标组"""
    line = (
        "验证集accuracy/top1: 68.5415  single-label/precision: 62.2472  "
        "single-label/recall: 56.6112  single-label/f1-score: 56.5116"
    )
    result = parse_metric_line(line)
    assert len(result) == 4
    assert result["验证集accuracy"] == 68.5415
    assert result["验证集precision"] == 62.2472
    assert result["验证集recall"] == 56.6112
    assert result["验证集f1"] == 56.5116


def test_parse_metric_line_with_default_group():
    """指定 default_group 时覆盖自动检测的组前缀"""
    line = "top1: 90.12  precision: 88.34  recall: 85.67  f1: 86.98"
    result = parse_metric_line(line, default_group="Val")
    assert result["Valaccuracy"] == 90.12
    assert result["Valprecision"] == 88.34
    assert result["Valrecall"] == 85.67
    assert result["Valf1"] == 86.98


def test_parse_metric_line_with_percent_values():
    """带百分号的值正常转为数字"""
    line = "测试集accuracy: 78.50%  precision: 75.30%  recall: 72.10%  f1: 73.60%"
    result = parse_metric_line(line)
    assert result["测试集accuracy"] == 78.50
    assert result["测试集precision"] == 75.30
    assert result["测试集recall"] == 72.10
    assert result["测试集f1"] == 73.60


def test_parse_metric_line_single_token_returns_empty():
    """单 token 无法构成复合指标行，返回空"""
    assert parse_metric_line("Dice = 0.87") == {}
    assert parse_metric_line("accuracy: 95.0") == {}


def test_parse_metrics_text_mixed_format():
    """支持复合格式与简单 key=value 混用"""
    text = """\
验证集accuracy/top1: 91.2  precision: 87.3  recall: 84.1  f1: 85.5
loss = dice
augmentation = flip
"""
    result = parse_metrics_text(text)
    assert result["验证集accuracy"] == 91.2
    assert result["验证集precision"] == 87.3
    assert result["验证集recall"] == 84.1
    assert result["验证集f1"] == 85.5
    assert result["loss"] == "dice"
    assert result["augmentation"] == "flip"


def test_parse_metrics_text_comment_lines_ignored():
    text = """\
# 这是注释
验证集accuracy/top1: 80.0  precision: 70.0  recall: 60.0  f1: 65.0
"""
    result = parse_metrics_text(text)
    assert "验证集accuracy" in result
    assert "#" not in str(result)


# ---------------------------------------------------------------------------
# to_float with percent
# ---------------------------------------------------------------------------


def test_to_float_handles_percent():
    assert to_float("69.84%") == 69.84
    assert to_float("100%") == 100.0
    assert to_float("0.5%") == 0.5
    assert to_float(" 42 % ") == 42.0
    assert to_float(None) is None
    assert to_float("") is None


# ---------------------------------------------------------------------------
# pick_default_param_keys (enhanced priority)
# ---------------------------------------------------------------------------


def test_pick_default_param_keys_lr_first():
    """学习率、优化器、调度器应排在前面"""
    keys = [
        "work_dir",
        "train_dataloader.batch_size",
        "optim_wrapper.optimizer.lr",
        "optim_wrapper.optimizer.type",
        "param_scheduler.type",
        "model.type",
        "dataset_type",
    ]
    selected = pick_default_param_keys(keys)
    # lr 必须在最前面
    assert selected[0] == "optim_wrapper.optimizer.lr"
    assert selected[1] == "optim_wrapper.optimizer.type"
    assert "param_scheduler.type" in selected[:4]


def test_pick_default_param_keys_lr_variants():
    """多种学习率/优化器 key 变体都被优先选中"""
    keys = [
        "unknown_key",
        "lr",
        "learning_rate",
        "base_lr",
        "optimizer",
        "lr_scheduler",
    ]
    selected = pick_default_param_keys(keys)
    assert selected[0] == "lr"
    assert selected[1] == "learning_rate"
    assert selected[2] == "base_lr"
    assert selected[3] == "optimizer"
    assert "lr_scheduler" in selected


# ---------------------------------------------------------------------------
# extract_pipeline_summaries
# ---------------------------------------------------------------------------


def test_extract_pipeline_summaries():
    params = {
        "train_pipeline[0].type": "LoadImageFromFile",
        "train_pipeline[1].type": "RandomResizedCrop",
        "train_pipeline[1].scale": 224,
        "train_pipeline[2].type": "RandomFlip",
        "train_pipeline[3].type": "PackClsInputs",
        "test_pipeline[0].type": "LoadImageFromFile",
        "test_pipeline[1].type": "ResizeEdge",
        "test_pipeline[2].type": "CenterCrop",
        "test_pipeline[3].type": "PackClsInputs",
        "model.type": "ImageClassifier",
    }
    summaries = extract_pipeline_summaries(params)
    assert summaries["train_pipeline_summary"] == (
        "LoadImageFromFile → RandomResizedCrop → RandomFlip → PackClsInputs"
    )
    assert summaries["test_pipeline_summary"] == (
        "LoadImageFromFile → ResizeEdge → CenterCrop → PackClsInputs"
    )
    assert "val_pipeline_summary" not in summaries


# ---------------------------------------------------------------------------
# ConfigParser registry
# ---------------------------------------------------------------------------


def test_mmengine_parser_is_registered_by_default():
    """The built-in MMEngineConfigParser is auto-registered at import time."""
    from masr.config_parser import MMEngineConfigParser, get_registered_parsers

    parsers = get_registered_parsers()
    types = [type(p) for p in parsers]
    assert MMEngineConfigParser in types


def test_register_custom_parser():
    """A custom ConfigParser can be registered and appears in the registry."""
    from masr.config_parser import ConfigParser, register_parser, get_registered_parsers

    class FakeParser(ConfigParser):
        def parse(self, text):
            return {"custom_key": text.strip()}

    count_before = len(get_registered_parsers())
    register_parser(FakeParser())
    assert len(get_registered_parsers()) == count_before + 1
    # Verify the newly registered parser is of the correct type.
    assert isinstance(get_registered_parsers()[-1], FakeParser)


def test_extract_params_fallback_uses_next_parser():
    """When one parser raises SyntaxError, the next registered parser is tried.

    The auto-registered MMEngineConfigParser raises SyntaxError on invalid
    Python; a previously registered custom parser (from
    ``test_register_custom_parser``) handles the text instead.
    """
    from masr.config_parser import extract_params_from_config_text

    result = extract_params_from_config_text("not valid python {{{")
    # MMEngine fails → custom FakeParser succeeds, returning ``custom_key``.
    assert "custom_key" in result


# ---------------------------------------------------------------------------
# Log parser (extract_best_metrics)
# ---------------------------------------------------------------------------


def test_extract_best_metrics_from_mmengine_log():
    """Parse MMEngine-style training log with Epoch(val) markers and timestamps."""
    from masr.log_parser import extract_best_metrics

    log = """
2024/01/15 10:30:00 - mmengine - INFO - Epoch(val) [1][50/100]  lr: 0.001  loss: 0.5
2024/01/15 10:31:00 - mmengine - INFO - accuracy/top1: 68.5  single-label/precision: 62.2  single-label/recall: 56.6  single-label/f1-score: 56.5
2024/01/15 10:32:00 - mmengine - INFO - Epoch(val) [2][50/100]  lr: 0.0009  loss: 0.3
2024/01/15 10:33:00 - mmengine - INFO - accuracy/top1: 78.0  single-label/precision: 72.1  single-label/recall: 66.3  single-label/f1-score: 69.0
"""
    best, epoch, summary = extract_best_metrics(log)
    assert epoch == 2
    # Epoch(val) → "验证集" prefix
    assert best["验证集accuracy"] == 78.0
    assert best["验证集precision"] == 72.1
    assert best["验证集recall"] == 66.3
    assert best["验证集f1"] == 69.0
    assert "Epoch 2" in summary


def test_extract_best_metrics_simple_kv_format():
    """Parse colon-separated metric lines with 'val' group prefix."""
    from masr.log_parser import extract_best_metrics

    log = """
Epoch 1/10
val/accuracy: 0.72  val/loss: 0.35
Epoch 2/10
val/accuracy: 0.85  val/loss: 0.22
Epoch 3/10
val/accuracy: 0.81  val/loss: 0.19
"""
    best, epoch, summary = extract_best_metrics(log)
    assert epoch == 2  # highest accuracy
    # "val" prefix normalised to Chinese "验证集"
    assert best["验证集accuracy"] == 0.85
    assert best["验证集loss"] == 0.22


def test_extract_best_metrics_target_loss():
    """When target_metric is specified, find best epoch by that metric."""
    from masr.log_parser import extract_best_metrics

    log = """
Epoch 1/10
val/accuracy: 0.72  val/loss: 0.35
Epoch 2/10
val/accuracy: 0.85  val/loss: 0.22
Epoch 3/10
val/accuracy: 0.81  val/loss: 0.19
"""
    best, epoch, summary = extract_best_metrics(log, target_metric="val/loss")
    assert epoch == 3  # lowest loss
    assert best["验证集loss"] == 0.19


def test_extract_best_metrics_ignores_config_dump():
    """Config dumps (key = value) should be ignored; only colon metrics count."""
    from masr.log_parser import extract_best_metrics

    log = """
backbone.type = ResNet
optimizer.lr = 0.001
Epoch 1
accuracy: 72.5  loss: 0.35
train_cfg.max_epochs = 100
Epoch 2
accuracy: 85.0  loss: 0.22
"""
    best, epoch, summary = extract_best_metrics(log)
    assert epoch == 2  # highest accuracy
    assert best["accuracy"] == 85.0
    assert best["loss"] == 0.22
    # Config keys should NOT appear
    assert "backbone.type" not in best
    assert "optimizer.lr" not in best


def test_extract_best_metrics_filters_tensor_and_timing():
    """Confusion matrix dumps and data_time/time should be filtered out."""
    from masr.log_parser import extract_best_metrics

    log = """\
2026/06/11 15:33:35 - mmengine - INFO - Epoch(val) [141][66/66]    accuracy/top1: 66.2536  single-label/precision: 62.4004  single-label/recall: 61.9547  single-label/f1-score: 60.8335  confusion_matrix/result:
tensor([[ 61,  10,   5,   3,   0,   0],
        [ 24,  19,  19,   2,   1,   0],
        [  5,  11,  58,  15,   0,   0],
        [  4,   4,  74, 269,  25,   0],
        [  0,   0,   2,  84, 207,  15],
        [  0,   0,   3,   2,  46,  81]])
  data_time: 0.0023  time: 0.0816"""
    best, epoch, summary = extract_best_metrics(log)
    assert epoch == 141
    # Epoch(val) → "验证集" prefix
    assert best["验证集accuracy"] == 66.2536
    assert best["验证集precision"] == 62.4004
    assert best["验证集recall"] == 61.9547
    assert best["验证集f1"] == 60.8335
    # Non-metric keys must be absent.
    assert "data_time" not in best
    assert "time" not in best
    assert "confusion_matrix" not in best


def test_extract_best_metrics_no_eval_blocks():
    """Return empty dict when no evaluation blocks are found."""
    from masr.log_parser import extract_best_metrics

    best, epoch, summary = extract_best_metrics("just some random text\nnothing here\n")
    assert best == {}
    assert epoch == 0


# ---------------------------------------------------------------------------
# Log parser (extract_epoch_curves)
# ---------------------------------------------------------------------------


def test_extract_epoch_curves_from_mmengine_log():
    """Extract per-epoch validation metrics from MMEngine training log."""
    from masr.log_parser import extract_epoch_curves

    log = """
2024/01/15 10:30:00 - mmengine - INFO - Epoch(val) [1][50/100]    accuracy/top1: 12.30  single-label/precision: 3.68  single-label/recall: 16.71  single-label/f1-score: 4.94
2024/01/15 10:31:00 - mmengine - INFO - Epoch(val) [2][50/100]    accuracy/top1: 45.00  single-label/precision: 15.00  single-label/recall: 35.00  single-label/f1-score: 21.00
2024/01/15 10:32:00 - mmengine - INFO - Epoch(val) [3][50/100]    accuracy/top1: 68.50  single-label/precision: 62.25  single-label/recall: 56.60  single-label/f1-score: 59.20
"""
    curves = extract_epoch_curves(log)

    assert curves["epoch"] == [1, 2, 3]
    assert len(curves["验证集accuracy"]) == 3
    assert curves["验证集accuracy"] == [12.30, 45.00, 68.50]
    assert curves["验证集f1"] == [4.94, 21.00, 59.20]
    assert curves["验证集precision"] == [3.68, 15.00, 62.25]
    assert curves["验证集recall"] == [16.71, 35.00, 56.60]


def test_extract_epoch_curves_empty_log():
    """Return empty dict for logs without Epoch(val) lines."""
    from masr.log_parser import extract_epoch_curves

    curves = extract_epoch_curves("just some random text\nnothing here\n")
    assert curves == {}


def test_extract_epoch_curves_filters_sporadic_metrics():
    """Only keep metrics that appear in ≥50% of epochs."""
    from masr.log_parser import extract_epoch_curves

    log = """
2024/01/15 10:30:00 - mmengine - INFO - Epoch(val) [1][50/100]    accuracy/top1: 12.30  single-label/f1-score: 4.94
2024/01/15 10:31:00 - mmengine - INFO - Epoch(val) [2][50/100]    accuracy/top1: 45.00  single-label/f1-score: 21.00  single-label/precision: 15.00
2024/01/15 10:32:00 - mmengine - INFO - Epoch(val) [3][50/100]    accuracy/top1: 68.50  single-label/f1-score: 59.20
"""
    curves = extract_epoch_curves(log)

    # accuracy and f1 appear in 3/3 epochs → kept; precision only in 1/3 → dropped
    assert "验证集accuracy" in curves
    assert "验证集f1" in curves
    assert "验证集precision" not in curves


def test_extract_epoch_curves_includes_train_loss():
    """Extract training loss from Epoch(train) lines alongside val metrics."""
    from masr.log_parser import extract_epoch_curves

    log = """
2024/01/15 10:30:00 - mmengine - INFO - Epoch(train) [1][50/100]  loss: 1.1556
2024/01/15 10:31:00 - mmengine - INFO - Epoch(val) [1][50/100]    accuracy/top1: 12.30  single-label/f1-score: 4.94
2024/01/15 10:32:00 - mmengine - INFO - Epoch(train) [2][50/100]  loss: 0.8923
2024/01/15 10:33:00 - mmengine - INFO - Epoch(val) [2][50/100]    accuracy/top1: 45.00  single-label/f1-score: 21.00
2024/01/15 10:34:00 - mmengine - INFO - Epoch(train) [3][50/100]  loss: 0.5432
2024/01/15 10:35:00 - mmengine - INFO - Epoch(val) [3][50/100]    accuracy/top1: 68.50  single-label/f1-score: 59.20
"""
    curves = extract_epoch_curves(log)

    assert curves["epoch"] == [1, 2, 3]
    assert "训练集loss" in curves
    assert "验证集accuracy" in curves
    assert "验证集f1" in curves
    assert curves["训练集loss"] == [1.1556, 0.8923, 0.5432]
    assert curves["验证集accuracy"] == [12.30, 45.00, 68.50]
    assert curves["验证集f1"] == [4.94, 21.00, 59.20]
