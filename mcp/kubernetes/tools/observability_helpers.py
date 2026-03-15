import json
import os
from typing import Dict, List
from urllib.request import Request, urlopen


DEFAULT_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:32000")
DEFAULT_JAEGER_URL = os.getenv("JAEGER_URL", "http://127.0.0.1:30462")


def http_json(url: str, timeout: int = 30) -> Dict[str, object]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def read_csv(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"error: file '{file_path}' not found."
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        return f"error: failed to read '{file_path}': {exc}"


def greedy_compress_lines(text: str, max_lines: int = 500) -> str:
    lines = text.splitlines()
    compressed: List[str] = []
    last_line: str = ""
    repeat_count = 0

    for line in lines:
        if line == last_line:
            repeat_count += 1
            compressed[-1] = f"{line} (x{repeat_count + 1})"
        else:
            repeat_count = 0
            compressed.append(line)
            last_line = line

        if len(compressed) >= max_lines:
            compressed.append("... (truncated)")
            break

    return "\n".join(compressed)
