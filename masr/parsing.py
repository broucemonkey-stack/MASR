from __future__ import annotations

import ast
import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Composite metric-line parser
# ---------------------------------------------------------------------------

# Matches a single "key : number" token inside a composite metric line.
# Keys may contain Chinese characters, letters, digits, hyphens, underscores,
# dots, and slashes.  Values are integer or decimal numbers, optionally
# followed by a percent sign.
_METRIC_TOKEN_RE = re.compile(
    r"([^\s:]+)\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\s*%?)"
)

# Normalised metric-name *stems* — when a sub-key ends with one of these
# (ignoring case / underscore-vs-hyphen) we recognise it as a metric.
_METRIC_STEMS: set[str] = {
    "accuracy",
    "top1",
    "top-1",
    "top5",
    "top-5",
    "precision",
    "recall",
    "f1",
    "f1-score",
    "f1_score",
    "loss",
    "auc",
    "map",
    "map50",
    "map50-95",
    "iou",
    "miou",
    "dice",
    "hd95",
}

# Standardise common metric-name variants.
_METRIC_NAME_NORMALIZE: dict[str, str] = {
    "top1": "accuracy",
    "top-1": "accuracy",
    "top5": "accuracy@5",
    "top-5": "accuracy@5",
    "f1-score": "f1",
    "f1_score": "f1",
}

# Known group prefixes (longest first so we match greedily).
_GROUP_PREFIXES: list[str] = sorted(
    [
        "验证集",
        "测试集",
        "训练集",
        "validation",
        "training",
        "testing",
        "val",
        "test",
        "train",
    ],
    key=len,
    reverse=True,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_tags(value: str) -> list[str]:
    return [item.strip() for item in value.replace("，", ",").split(",") if item.strip()]


def parse_key_value_lines(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = coerce_scalar(value)
    return result


def parse_metric_line(text: str, default_group: str = "") -> dict[str, Any]:
    """Parse a *single* composite metric line from MMEngine / OpenMMLab logs.

    Typical input::

        验证集accuracy/top1: 68.5415  single-label/precision: 62.2472
        single-label/recall: 56.6112  single-label/f1-score: 56.5116

    Returns a flat dict whose keys are ``{group}{normalised_metric}``::

        {
            "验证集accuracy": 68.5415,
            "验证集precision": 62.2472,
            "验证集recall": 56.6112,
            "验证集f1": 56.5116,
        }

    If *default_group* is given it takes precedence over auto-detection.
    Returns an empty dict when fewer than 2 metric tokens are found.
    """
    matches = _METRIC_TOKEN_RE.findall(text)
    if len(matches) < 2:
        return {}

    group = default_group or _detect_group_prefix(matches[0][0])

    result: dict[str, Any] = {}
    for raw_key, raw_value in matches:
        metric_name = _extract_metric_name(raw_key)
        if metric_name is None:
            continue
        value = coerce_scalar(raw_value.strip().rstrip("%"))
        result[f"{group}{metric_name}"] = value

    return result


def parse_metrics_text(text: str) -> dict[str, Any]:
    """Parse arbitrary metrics text that may mix simple ``key=value`` lines
    with composite MMEngine-style metric lines.

    This is the recommended entry-point for the metrics textarea in the UI.
    """
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        composite = parse_metric_line(line)
        if composite:
            result.update(composite)
        else:
            result.update(parse_key_value_lines(line))
    return result


def coerce_scalar(value: str) -> Any:
    """Convert a string value to the most specific Python scalar."""
    value = value.strip().rstrip("%")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if _looks_like_literal(value):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    try:
        if value and not any(char in value for char in [".", "e", "E"]):
            return int(value)
        return float(value)
    except ValueError:
        return value


def format_key_value_lines(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, str):
            rendered = value
        else:
            rendered = repr(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers (metric-line)
# ---------------------------------------------------------------------------


def _detect_group_prefix(first_key: str) -> str:
    """Guess the evaluation-set prefix (e.g. ``验证集``) from the first
    token of a composite metric line."""
    for prefix in _GROUP_PREFIXES:
        if first_key.startswith(prefix):
            return prefix
    # Fallback: split before the first recognised metric stem.
    key_lower = first_key.lower().replace("_", "-")
    for stem in sorted(_METRIC_STEMS, key=len, reverse=True):
        stem_norm = stem.replace("_", "-")
        idx = key_lower.find(stem_norm)
        if idx > 0:
            return first_key[:idx].rstrip("/-_")
    return ""


def _extract_metric_name(key: str) -> str | None:
    """Isolate the metric name (e.g. ``accuracy``, ``f1``) from a composite
    key like ``验证集accuracy/top1`` or ``single-label/precision``."""
    # Strip known group prefix.
    remaining = key
    for prefix in _GROUP_PREFIXES:
        if remaining.startswith(prefix):
            remaining = remaining[len(prefix) :]
            break

    parts = [p.strip() for p in remaining.split("/") if p.strip()]
    if not parts:
        return None

    # Walk parts right-to-left looking for a recognised stem.
    for part in reversed(parts):
        part_norm = part.lower().replace("_", "-")
        if part_norm in _METRIC_STEMS:
            return _METRIC_NAME_NORMALIZE.get(part_norm, part)

    # Last resort: return the final path component, normalised.
    metric = parts[-1]
    part_norm = metric.lower().replace("_", "-")
    return _METRIC_NAME_NORMALIZE.get(part_norm, metric)


def _looks_like_literal(value: str) -> bool:
    value = value.strip()
    return (
        len(value) >= 2
        and value[0] in {"[", "{", "(", "'", '"'}
        and value[-1] in {"]", "}", ")", "'", '"'}
    )
