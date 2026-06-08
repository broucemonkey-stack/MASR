from __future__ import annotations

import ast
from typing import Any


def extract_params_from_config_text(text: str) -> dict[str, Any]:
    """Extract flattened key/value parameters from a Python config file.

    The parser is intentionally safe: it reads Python syntax with ``ast`` and
    evaluates only literals plus ``dict(...)`` calls commonly used by MMEngine
    configs. It never imports or executes the uploaded config.
    """

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
    return flattened


def pick_default_param_keys(keys: list[str], limit: int = 12) -> list[str]:
    priority = [
        "dataset_type",
        "model.type",
        "model.backbone.type",
        "model.backbone.depth",
        "model.head.type",
        "model.head.num_classes",
        "model.head.loss.type",
        "optim_wrapper.optimizer.type",
        "optim_wrapper.optimizer.lr",
        "optim_wrapper.optimizer.weight_decay",
        "train_cfg.max_epochs",
        "train_cfg.val_interval",
        "train_dataloader.batch_size",
        "val_dataloader.batch_size",
        "test_dataloader.batch_size",
        "param_scheduler.type",
        "work_dir",
    ]
    selected = [key for key in priority if key in keys]
    for key in keys:
        if len(selected) >= limit:
            break
        if key not in selected:
            selected.append(key)
    return selected


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
