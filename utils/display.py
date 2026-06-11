"""Display-oriented utilities for the MASR Streamlit app.

These helpers bridge the gap between :mod:`masr` data models and the
pandas / Streamlit rendering layer.  Keep them free of Streamlit imports
so they remain testable without a Streamlit runtime.
"""

from __future__ import annotations

from typing import Any

from masr.filters import collect_dynamic_keys
from masr.models import Experiment


def _clean_number(value: Any) -> str:
    """Convert a value to a clean string with no trailing zeros.

    Floats use :g formatting (strips trailing zeros), ints stay as-is,
    everything else is stringified.
    """
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def flatten_experiments(
    experiments: list[Experiment],
    param_keys: list[str] | None = None,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten a list of Experiment objects into rows for a DataFrame."""
    param_keys = param_keys if param_keys is not None else collect_dynamic_keys(experiments, "params")
    metric_keys = metric_keys if metric_keys is not None else collect_dynamic_keys(experiments, "metrics")
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        row: dict[str, Any] = {
            "实验名称": experiment.name,
            "数据集": experiment.dataset,
            "模型": experiment.model,
        }
        for key in metric_keys:
            row[f"指标:{key}"] = _clean_number(experiment.metrics.get(key, ""))
        row.update({
            "策略": experiment.strategy,
            "随机种子": experiment.seed,
            "标签": ", ".join(experiment.tags),
            "更新时间": experiment.updated_at,
        })
        for key in param_keys:
            row[f"参数:{key}"] = _clean_number(experiment.params.get(key, ""))
        rows.append(row)
    return rows


def experiment_label(experiment: Experiment) -> str:
    """Build a human-readable label for an experiment."""
    parts = [experiment.name]
    detail = " / ".join(part for part in [experiment.dataset, experiment.model, experiment.strategy] if part)
    if detail:
        parts.append(detail)
    return " · ".join(parts)
