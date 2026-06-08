from __future__ import annotations

import ast
import json
from typing import Any


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


def coerce_scalar(value: str) -> Any:
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


def _looks_like_literal(value: str) -> bool:
    value = value.strip()
    return (
        len(value) >= 2
        and value[0] in {"[", "{", "(", "'", '"'}
        and value[-1] in {"]", "}", ")", "'", '"'}
    )
