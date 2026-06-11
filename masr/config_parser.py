from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Plugin infrastructure
# ---------------------------------------------------------------------------


class ConfigParser(ABC):
    """Abstract base class for configuration file parsers.

    Subclass and register via :func:`register_parser` to add support for
    new config formats.
    """

    @abstractmethod
    def parse(self, text: str) -> dict[str, Any]:
        """Parse configuration *text* into a flat key/value dict.

        Raises:
            SyntaxError: if *text* is syntactically invalid for this parser.
        """
        ...


_registry: list[ConfigParser] = []


def register_parser(parser: ConfigParser) -> None:
    """Register a configuration parser for auto-detection.

    Parsers are tried in registration order; the first successful result
    is returned by :func:`extract_params_from_config_text`.
    """
    _registry.append(parser)


def get_registered_parsers() -> list[ConfigParser]:
    """Return a copy of the current parser registry."""
    return list(_registry)


# ---------------------------------------------------------------------------
# MMEngine-style Python config parser
# ---------------------------------------------------------------------------


class MMEngineConfigParser(ConfigParser):
    """Parser for MMEngine / OpenMMLab Python configuration files.

    Safely reads Python syntax with ``ast`` and evaluates only literals
    plus ``dict(...)`` calls.  It never imports or executes the uploaded
    config.
    """

    def parse(self, text: str) -> dict[str, Any]:
        tree = ast.parse(text)
        assignments: dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = _safe_eval(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assignments[node.target.id] = _safe_eval(node.value)

        flattened: dict[str, Any] = {}
        for key, value in assignments.items():
            _flatten_value(key, value, flattened)

        # Append human-readable pipeline summaries.
        flattened.update(extract_pipeline_summaries(flattened))
        return flattened


# ---------------------------------------------------------------------------
# Public entry point (delegates to registered parsers)
# ---------------------------------------------------------------------------


def extract_params_from_config_text(text: str) -> dict[str, Any]:
    """Extract flattened key/value parameters from a Python config file.

    Delegates to registered :class:`ConfigParser` instances in order;
    returns the result from the first parser that succeeds.

    The parser is intentionally safe: it never imports or executes the
    uploaded config.
    """
    errors: list[SyntaxError] = []
    for parser in _registry:
        try:
            return parser.parse(text)
        except SyntaxError as exc:
            errors.append(exc)
            continue

    if errors:
        raise errors[0]
    return {}


def pick_default_param_keys(keys: list[str], limit: int = 12) -> list[str]:
    """Return a curated subset of config keys for display.

    Priority order (highest first)::

        1. Learning rate      — optim_wrapper.optimizer.lr / lr / …
        2. Optimizer           — optim_wrapper.optimizer.type
        3. LR scheduler        — param_scheduler …
        4. Train pipeline      — train_pipeline / train_dataloader …
        5. Test pipeline       — test_pipeline / val_pipeline / test_dataloader …
        6. Other common keys   — model, batch_size, max_epochs, …
    """
    priority = [
        # ---- 学习率 (learning rate) ----
        "optim_wrapper.optimizer.lr",
        "lr",
        "learning_rate",
        "base_lr",
        # ---- 优化器 (optimizer) ----
        "optim_wrapper.optimizer.type",
        "optimizer.type",
        "optimizer",
        "optim_wrapper.optimizer.weight_decay",
        # ---- 学习率衰减 / 调度器 (LR scheduler / decay) ----
        "param_scheduler.type",
        "param_scheduler",
        "lr_scheduler",
        "lr_decay",
        "lr_schedule",
        # ---- train_pipeline ----
        "train_pipeline",
        "train_pipeline_summary",
        "train_dataloader.dataset.pipeline",
        "train_dataloader.batch_size",
        # ---- test_pipeline ----
        "test_pipeline",
        "test_pipeline_summary",
        "val_pipeline",
        "val_pipeline_summary",
        "test_dataloader.dataset.pipeline",
        "val_dataloader.dataset.pipeline",
        "test_dataloader.batch_size",
        "val_dataloader.batch_size",
        # ---- 通用 / 其他常用 ----
        "dataset_type",
        "model.type",
        "model.backbone.type",
        "model.backbone.depth",
        "model.backbone.arch",
        "model.head.type",
        "model.head.num_classes",
        "model.head.loss.type",
        "train_cfg.max_epochs",
        "train_cfg.val_interval",
        "work_dir",
    ]
    selected = [key for key in priority if key in keys]
    for key in keys:
        if len(selected) >= limit:
            break
        if key not in selected:
            selected.append(key)
    return selected


def extract_pipeline_summaries(params: dict[str, Any]) -> dict[str, str]:
    """Post-process extracted config params and create human-readable pipeline
    summaries.

    MMEngine configs often define pipelines as lists of ``dict(type=…)``
    calls.  After ``extract_params_from_config_text`` each step becomes
    ``train_pipeline[0].type``, ``train_pipeline[1].type``, … which is
    verbose.  This helper collects those into compact arrow-chain strings::

        {"train_pipeline_summary": "LoadImageFromFile → RandomResizedCrop → …"}
    """
    summaries: dict[str, str] = {}
    for candidate in ("train_pipeline", "test_pipeline", "val_pipeline"):
        prefix = candidate + "["
        indices: dict[int, str] = {}
        for key, value in params.items():
            if key.startswith(prefix) and key.endswith(".type"):
                try:
                    idx = int(key[len(prefix) : key.index("]")])
                except (ValueError, IndexError):
                    continue
                indices[idx] = str(value)
        if indices:
            chain = " → ".join(v for _, v in sorted(indices.items()))
            summaries[f"{candidate}_summary"] = chain

    return summaries


def _safe_eval(node: ast.AST | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(item) for item in node.elts)
    if isinstance(node, ast.Set):
        return sorted(_safe_eval(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            str(_safe_eval(key)): _safe_eval(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        if isinstance(value, (int, float)):
            return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and _is_dict_call(node):
        payload: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is not None:
                payload[keyword.arg] = _safe_eval(keyword.value)
        return payload
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _is_dict_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args


def _flatten_value(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if not value:
            output[prefix] = {}
            return
        for key, child in value.items():
            _flatten_value(f"{prefix}.{key}", child, output)
        return

    if isinstance(value, (list, tuple)):
        if _is_scalar_sequence(value):
            output[prefix] = list(value)
            return
        if not value:
            output[prefix] = []
            return
        for index, child in enumerate(value):
            _flatten_value(f"{prefix}[{index}]", child, output)
        return

    output[prefix] = value


def _is_scalar_sequence(value: list[Any] | tuple[Any, ...]) -> bool:
    return all(not isinstance(item, (dict, list, tuple)) for item in value)


# ---------------------------------------------------------------------------
# Auto-register built-in parser
# ---------------------------------------------------------------------------
register_parser(MMEngineConfigParser())
