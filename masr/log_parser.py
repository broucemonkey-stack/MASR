"""Extract best-epoch validation metrics from MMEngine training logs.

Supports two strategies (tried in order):

1. **Best-checkpoint line** — the log contains lines like
   ``The best checkpoint with 60.8335 single-label/f1-score at 141 epoch``.
   We find the *last* such line, identify its epoch, and extract **only
   validation-set** metrics from the matching ``Epoch(val)`` line.

2. **Full-scan fallback** — when no checkpoint line is found, all
   ``Epoch(val)`` / ``Epoch(test)`` blocks are parsed and the best epoch
   is selected via *target_metric* (or auto-detected).

Non-metric noise (``data_time``, ``time``, ``base_lr``, ``eta``,
confusion-matrix dumps, timestamps, …) is filtered out.
"""

from __future__ import annotations

import re
from typing import Any

from .parsing import METRIC_TOKEN_RE, parse_metric_line, parse_metrics_text

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s*(?:\((\w+)\))?\s*\[?(\d+)\]?",
    re.IGNORECASE,
)

# "The best checkpoint with 60.8335 single-label/f1-score at 141 epoch"
_BEST_CKPT_RE = re.compile(
    r"The best checkpoint with\s+([0-9]+(?:\.[0-9]+)?)\s+(\S+)\s+at\s+(\d+)\s+epoch",
    re.IGNORECASE,
)

_TENSOR_LINE_RE = re.compile(
    r"^\s*(?:tensor\(\[|\[\[|[\]\),\s\d.\-eE]+$)"
)

# ---------------------------------------------------------------------------
# Group-label normalisation
# ---------------------------------------------------------------------------

_EPOCH_GROUP_MAP: dict[str, str] = {
    "val": "验证集",
    "validation": "验证集",
    "test": "测试集",
    "testing": "测试集",
    "train": "训练集",
    "training": "训练集",
}

# Training-set group labels — these are excluded from best-epoch selection.
_TRAIN_GROUPS: set[str] = {"训练集", "train", "training"}

# Evaluation-set group labels.
_EVAL_GROUPS: set[str] = {"验证集", "测试集", "val", "validation", "test", "testing"}

# ---------------------------------------------------------------------------
# Keys that are never evaluation metrics
# ---------------------------------------------------------------------------

_NON_METRIC_KEYS: set[str] = {
    "data_time",
    "time",
    "memory",
    "lr",
    "epoch",
    "iter",
    "base_lr",
    "eta",
}

# Key stems whose value is not a scalar metric (matrix dumps, etc.).
_NON_METRIC_STEMS: set[str] = {
    "confusion_matrix",
    "confusion_matrix/result",
    "cm",
}

# ---------------------------------------------------------------------------
# Metric-name stems for "best epoch" selection
# ---------------------------------------------------------------------------

_HIGHER_BETTER: set[str] = {
    "accuracy",
    "f1",
    "f1-score",
    "f1_score",
    "precision",
    "recall",
    "auc",
    "map",
    "map50",
    "map50-95",
    "iou",
    "miou",
    "dice",
}

_LOWER_BETTER: set[str] = {
    "loss",
    "error",
    "hd95",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_best_metrics(
    log_text: str,
    target_metric: str | None = None,
) -> tuple[dict[str, Any], int, str]:
    """Scan *log_text* and return the metrics from the best epoch.

    Strategy:
        1. If the log contains "best checkpoint" lines, use the **last**
           one to determine the best epoch, then extract its ``Epoch(val)``
           metrics.
        2. Otherwise scan all eval blocks and pick the best via
           *target_metric*.

    Parameters
    ----------
    log_text : str
        Full training log text.
    target_metric : str | None
        Metric name to optimise (e.g. ``"验证集f1"`` or ``"val/loss"``).
        Auto-detected when omitted.

    Returns
    -------
    (best_metrics, best_epoch, summary)
    """
    # --- strategy 1: best-checkpoint line -----------------------------------
    ckpt_epoch = _find_best_checkpoint_epoch(log_text)
    if ckpt_epoch is not None:
        val_metrics = _extract_val_metrics_for_epoch(log_text, ckpt_epoch)
        if val_metrics:
            summary = _build_summary(val_metrics, ckpt_epoch)
            return val_metrics, ckpt_epoch, summary

    # --- strategy 2: full-scan ----------------------------------------------
    blocks = _find_eval_blocks(log_text)
    if not blocks:
        return {}, 0, "未在日志中找到评估指标"

    parsed: list[tuple[int, dict[str, Any]]] = []
    for epoch, metrics in blocks:
        if metrics:
            parsed.append((epoch, metrics))

    if not parsed:
        return {}, 0, "未能解析任何评估指标"

    best_epoch, best_metrics = _pick_best(parsed, target_metric)
    summary = _build_summary(best_metrics, best_epoch)
    return best_metrics, best_epoch, summary


def extract_test_metrics(log_text: str) -> dict[str, Any]:
    """Extract metrics from ``Epoch(test)`` lines in a test-run log.

    Unlike training logs, test logs typically contain a single evaluation
    pass — no "best epoch" selection is needed.  All metrics from the
    ``Epoch(test)`` line are returned directly.

    Returns an empty dict when no test metrics are found.
    """
    lines = log_text.splitlines()
    current_epoch = 0
    current_group = ""
    current_raw_group = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _TENSOR_LINE_RE.match(stripped):
            continue

        m = _EPOCH_RE.search(stripped)
        if m:
            current_epoch = int(m.group(2))
            current_raw_group = (m.group(1) or "").lower()
            current_group = _EPOCH_GROUP_MAP.get(current_raw_group, current_raw_group)

        # Only process Epoch(test) lines.
        if current_raw_group not in {"test", "testing"}:
            continue

        metrics = parse_metric_line(stripped, default_group=current_group)
        if metrics:
            metrics = _filter_metrics(metrics)
            if metrics:
                return metrics

        # Fallback: mixed format.
        metrics = parse_metrics_text(stripped)
        if metrics:
            metrics = _filter_metrics(metrics)
            if metrics:
                return metrics

    return {}


def extract_epoch_curves(log_text: str) -> dict[str, list]:
    """Extract per-epoch metrics from both validation and training log lines.

    Scans ``Epoch(val)`` lines for evaluation metrics (accuracy, f1,
    precision, recall, …) and ``Epoch(train)`` lines for training metrics
    (loss, …).  Keys carry Chinese group prefixes — ``"验证集accuracy"``,
    ``"训练集loss"``, etc.

    Only metrics that appear in at least half of all epochs are kept —
    sporadic or malformed entries are discarded.

    Parameters
    ----------
    log_text : str
        Full training log text.

    Returns
    -------
    dict[str, list]
        Keys are ``"epoch"`` and metric names.  Values are equal-length
        lists ready for ``pd.DataFrame`` / ``st.plotly_chart()``.
    """
    lines = log_text.splitlines()
    current_epoch = 0
    current_raw_group = ""
    current_group = ""

    # epoch -> {metric_key: value}
    epoch_metrics: dict[int, dict[str, float]] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _TENSOR_LINE_RE.match(stripped):
            continue

        m = _EPOCH_RE.search(stripped)
        if m:
            current_epoch = int(m.group(2))
            current_raw_group = (m.group(1) or "").lower()
            current_group = _EPOCH_GROUP_MAP.get(current_raw_group, current_raw_group)

        # Process both validation and training lines.
        if current_raw_group not in {"val", "validation", "train", "training"}:
            continue

        metrics = parse_metric_line(stripped, default_group=current_group)
        if not metrics:
            # Fallback: mixed format.
            metrics = parse_metrics_text(stripped)

        metrics = _filter_metrics(metrics)
        if not metrics:
            continue

        # Convert to float and store (merge with any existing metrics for
        # the same epoch, e.g. val + train).
        float_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                float_metrics[k] = float(str(v).rstrip("%"))
            except (TypeError, ValueError):
                continue

        if float_metrics:
            if current_epoch not in epoch_metrics:
                epoch_metrics[current_epoch] = {}
            epoch_metrics[current_epoch].update(float_metrics)

    if not epoch_metrics:
        return {}

    # --- keep only metrics that appear in ≥50% of epochs ---------------------
    total_epochs = len(epoch_metrics)
    metric_keys: set[str] = set()
    for metrics in epoch_metrics.values():
        metric_keys.update(metrics.keys())

    common_keys: list[str] = []
    for key in sorted(metric_keys):
        count = sum(1 for m in epoch_metrics.values() if key in m)
        if count >= total_epochs * 0.5:
            common_keys.append(key)

    if not common_keys:
        return {}

    # --- build result dict ----------------------------------------------------
    sorted_epochs = sorted(epoch_metrics.keys())
    result: dict[str, list] = {"epoch": sorted_epochs}
    for key in common_keys:
        values: list[float | None] = []
        for ep in sorted_epochs:
            values.append(epoch_metrics[ep].get(key))
        result[key] = values  # type: ignore[assignment]

    return result


# ---------------------------------------------------------------------------
# Strategy 1: best-checkpoint line helpers
# ---------------------------------------------------------------------------


def _find_best_checkpoint_epoch(log_text: str) -> int | None:
    """Return the epoch number from the **last** best-checkpoint line.

    Returns ``None`` if no such line is found.
    """
    best_epoch: int | None = None
    for line in log_text.splitlines():
        m = _BEST_CKPT_RE.search(line)
        if m:
            best_epoch = int(m.group(3))
    return best_epoch


def _extract_val_metrics_for_epoch(
    log_text: str,
    target_epoch: int,
) -> dict[str, Any]:
    """Find the ``Epoch(val)`` line for *target_epoch* and parse its metrics.

    Returns an empty dict when the line cannot be found or parsed.
    """
    lines = log_text.splitlines()
    current_epoch = 0
    current_group = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _TENSOR_LINE_RE.match(stripped):
            continue

        # Update epoch / group.
        m = _EPOCH_RE.search(stripped)
        if m:
            current_epoch = int(m.group(2))
            raw_group = (m.group(1) or "").lower()
            current_group = _EPOCH_GROUP_MAP.get(raw_group, raw_group)

        if current_epoch != target_epoch:
            continue
        if raw_group in _TRAIN_GROUPS:
            continue

        # Try parse_metric_line first (composite format).
        metrics = parse_metric_line(stripped, default_group=current_group)
        if metrics:
            metrics = _filter_metrics(metrics)
            if metrics:
                return metrics

        # Fallback: parse_metrics_text for mixed formats.
        metrics = parse_metrics_text(stripped)
        if metrics:
            metrics = _filter_metrics(metrics)
            if metrics:
                return metrics

    return {}


# ---------------------------------------------------------------------------
# Strategy 2: full-scan helpers
# ---------------------------------------------------------------------------


def _find_eval_blocks(log_text: str) -> list[tuple[int, dict[str, Any]]]:
    """Find ``(epoch, metrics_dict)`` pairs from evaluation log lines.

    Only lines belonging to eval groups (val/test) are kept; training
    lines are skipped.
    """
    lines = log_text.splitlines()
    blocks: list[tuple[int, dict[str, Any]]] = []
    current_epoch = 0
    current_group = ""
    current_raw_group = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip tensor / confusion-matrix dump lines.
        if _TENSOR_LINE_RE.match(stripped):
            continue

        # Detect epoch number and optional group label (val/test/train).
        m = _EPOCH_RE.search(stripped)
        if m:
            current_epoch = int(m.group(2))
            current_raw_group = (m.group(1) or "").lower()
            current_group = _EPOCH_GROUP_MAP.get(current_raw_group, current_raw_group)

        # Skip training-only lines.
        if current_raw_group in _TRAIN_GROUPS:
            continue

        # Use parse_metric_line with the pre-detected group label.
        metrics = parse_metric_line(stripped, default_group=current_group)
        if not metrics:
            continue

        metrics = _filter_metrics(metrics)
        if not metrics:
            continue

        blocks.append((current_epoch, metrics))

    return blocks


def _pick_best(
    parsed: list[tuple[int, dict[str, Any]]],
    target_metric: str | None,
) -> tuple[int, dict[str, Any]]:
    """Select the best epoch from *parsed* based on *target_metric*."""
    if target_metric is None:
        all_keys: set[str] = set()
        for _, metrics in parsed:
            all_keys.update(metrics.keys())
        target_metric = _auto_target(all_keys)
    else:
        target_metric = _resolve_target_metric(target_metric, parsed)

    target_lower = _is_lower_better(target_metric)

    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_value: float | None = None

    for epoch, metrics in parsed:
        raw = metrics.get(target_metric)  # type: ignore[arg-type]
        if raw is None:
            continue
        try:
            value = float(str(raw).rstrip("%"))
        except (TypeError, ValueError):
            continue

        if best_value is None:
            best_value, best_epoch, best_metrics = value, epoch, metrics
        elif target_lower:
            if value < best_value:
                best_value, best_epoch, best_metrics = value, epoch, metrics
        else:
            if value > best_value:
                best_value, best_epoch, best_metrics = value, epoch, metrics

    return best_epoch, best_metrics


def _auto_target(keys: set[str]) -> str:
    """Pick the best metric name from a set of keys.

    Precedence: higher-is-better stems → lower-is-better stems → first
    sorted.  Both exact and substring matching against normalised stems
    are supported so that group-prefixed keys like ``验证集accuracy`` are
    recognised.
    """
    # Build lookup: normalised stem → original key(s)
    keys_norm: dict[str, list[str]] = {}
    for k in keys:
        nk = _normalise(k)
        keys_norm.setdefault(nk, []).append(k)
        # Also register sub-stems so that e.g. "accuracy" maps to "验证集accuracy".
        for stem in _HIGHER_BETTER | _LOWER_BETTER:
            if stem in nk and stem not in keys_norm:
                keys_norm.setdefault(stem, []).append(k)

    def _pick(stems: set[str]) -> str | None:
        for stem in stems:
            candidates = keys_norm.get(stem)
            if candidates:
                # Prefer val-group keys.
                for c in candidates:
                    if "验证集" in c or "val" in c.lower():
                        return c
                return candidates[0]
        return None

    result = _pick(_HIGHER_BETTER)
    if result:
        return result

    result = _pick(_LOWER_BETTER)
    if result:
        return result

    # Absolute fallback.
    return sorted(keys)[0] if keys else "accuracy"


def _resolve_target_metric(
    target: str,
    parsed: list[tuple[int, dict[str, Any]]],
) -> str:
    """Map a user-supplied target metric (e.g. ``"val/loss"``) to the
    canonical key used in the parsed metrics dict (e.g. ``"验证集loss"``).
    """
    # Already canonical — verify it exists.
    for _, metrics in parsed:
        if target in metrics:
            return target

    # Collect all known metric keys.
    all_keys: set[str] = set()
    for _, metrics in parsed:
        all_keys.update(metrics.keys())

    # --- try parse_metrics_text trick (old behaviour) ---
    if "/" in target:
        group_part = target.split("/")[0]
        fake = f"{group_part}/_x: 0  {target}: 1"
        canonical = parse_metrics_text(fake)
        for k in canonical:
            if "_x" not in k.lower() and k in all_keys:
                return k

    # --- fuzzy match: metric stem + any group ---
    target_norm = _normalise(target)
    for k in sorted(all_keys):
        if target_norm in _normalise(k):
            return k

    # --- reverse fuzzy: stem from target matches key ---
    # e.g. target="val/loss", key="验证集loss" → "loss" in both
    for stem in _HIGHER_BETTER | _LOWER_BETTER:
        if stem in target_norm:
            for k in all_keys:
                if stem in _normalise(k):
                    return k

    return target  # fallback


def _is_lower_better(metric: str) -> bool:
    """Return True if lower values of *metric* are better."""
    key = _normalise(metric)
    if key in _LOWER_BETTER:
        return True
    if "loss" in key or "error" in key:
        return True
    return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _filter_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove non-evaluation entries from a metrics dict."""
    result: dict[str, Any] = {}
    for k, v in metrics.items():
        if _is_eval_key(k):
            result[k] = v
    return result


def _is_eval_key(key: str) -> bool:
    """Return True if *key* looks like an evaluation metric name."""
    if key.lower() in _NON_METRIC_KEYS:
        return False
    if key.strip().isdigit():
        return False
    key_norm = _normalise(key)
    # Reject keys that are purely non-metric stems.
    for stem in _NON_METRIC_STEMS:
        if stem.replace("_", "").replace("/", "") in key_norm:
            return False
    if key_norm in _NON_METRIC_KEYS:
        return False
    # Must contain at least one known metric stem.
    known = _HIGHER_BETTER | _LOWER_BETTER
    return any(stem in key_norm for stem in known)


def _normalise(key: str) -> str:
    """Lower-case and strip separators for fuzzy matching."""
    return key.lower().replace("_", "").replace("-", "").replace("/", "").replace(" ", "")


def _build_summary(metrics: dict[str, Any], epoch: int) -> str:
    """Build a human-readable summary for the best epoch."""
    parts: list[str] = []
    for k, v in sorted(metrics.items()):
        try:
            fv = float(str(v).rstrip("%"))
            parts.append(f"{k} = {fv:.4f}")
        except (TypeError, ValueError):
            parts.append(f"{k} = {v}")
    joined = ", ".join(parts[:6])
    return f"Epoch {epoch} — {joined}"
