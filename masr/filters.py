from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Experiment


@dataclass(slots=True)
class ExperimentFilters:
    search: str = ""
    datasets: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    strategies: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    param_key: str = ""
    param_value: str = ""
    metric_key: str = ""
    metric_min: float | None = None
    metric_max: float | None = None


def filter_experiments(
    experiments: list[Experiment],
    filters: ExperimentFilters,
) -> list[Experiment]:
    return [experiment for experiment in experiments if matches_filters(experiment, filters)]


def matches_filters(experiment: Experiment, filters: ExperimentFilters) -> bool:
    if filters.search and filters.search.lower() not in _search_blob(experiment):
        return False
    if filters.datasets and experiment.dataset not in filters.datasets:
        return False
    if filters.models and experiment.model not in filters.models:
        return False
    if filters.strategies and experiment.strategy not in filters.strategies:
        return False
    if filters.tags and not filters.tags.issubset(set(experiment.tags)):
        return False
    if filters.param_key:
        value = experiment.params.get(filters.param_key)
        if value is None:
            return False
        if filters.param_value and filters.param_value.lower() not in str(value).lower():
            return False
    if filters.metric_key:
        value = experiment.metrics.get(filters.metric_key)
        if value is None:
            return False
        numeric_value = to_float(value)
        if filters.metric_min is not None or filters.metric_max is not None:
            if numeric_value is None:
                return False
            if filters.metric_min is not None and numeric_value < filters.metric_min:
                return False
            if filters.metric_max is not None and numeric_value > filters.metric_max:
                return False
    return True


def collect_values(experiments: list[Experiment], attr: str) -> list[str]:
    values = {str(getattr(experiment, attr, "")).strip() for experiment in experiments}
    return sorted(value for value in values if value)


def collect_tags(experiments: list[Experiment]) -> list[str]:
    tags: set[str] = set()
    for experiment in experiments:
        tags.update(tag for tag in experiment.tags if tag)
    return sorted(tags)


def collect_dynamic_keys(experiments: list[Experiment], field_name: str) -> list[str]:
    keys: set[str] = set()
    for experiment in experiments:
        payload = getattr(experiment, field_name, {})
        if isinstance(payload, dict):
            keys.update(str(key) for key in payload.keys())
    return sorted(keys)


def metric_range(experiments: list[Experiment], key: str) -> tuple[float, float] | None:
    values = [
        numeric
        for numeric in (to_float(experiment.metrics.get(key)) for experiment in experiments)
        if numeric is not None
    ]
    if not values:
        return None
    return min(values), max(values)


def to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _search_blob(experiment: Experiment) -> str:
    parts = [
        experiment.name,
        experiment.description,
        experiment.dataset,
        experiment.model,
        experiment.strategy,
        experiment.seed,
        " ".join(experiment.tags),
        " ".join(f"{key} {value}" for key, value in experiment.params.items()),
        " ".join(f"{key} {value}" for key, value in experiment.metrics.items()),
    ]
    return " ".join(parts).lower()
