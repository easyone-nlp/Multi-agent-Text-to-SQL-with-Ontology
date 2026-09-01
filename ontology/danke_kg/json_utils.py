from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(response: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.I | re.S)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("model 응답에서 JSON object를 찾지 못함")
