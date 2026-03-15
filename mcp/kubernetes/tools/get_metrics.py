import csv
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin

from .observability_helpers import DEFAULT_PROMETHEUS_URL, http_json

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def get_metrics(namespace: str, duration: int) -> str:
        """Collect metrics data from the service using Prometheus."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration)
        directory = tempfile.mkdtemp(prefix="metrics_")

        metrics = [
            "container_cpu_usage_seconds_total",
            "container_memory_working_set_bytes",
            "kube_pod_container_status_restarts_total",
        ]

        for metric in metrics:
            query = f'{metric}{{namespace="{namespace}"}}'
            params = {
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": max(int(duration * 60 / 60), 30),
            }
            url = urljoin(DEFAULT_PROMETHEUS_URL, "/api/v1/query_range") + "?" + urlencode(params)
            try:
                response = http_json(url)
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to query Prometheus for {metric}: {exc}") from exc

            if response.get("status") != "success":
                raise RuntimeError(f"Prometheus query failed for {metric}: {response}")

            values = response.get("data", {}).get("result", [])
            filename = os.path.join(directory, f"{metric}.csv")
            with open(filename, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["metric", "labels", "timestamp", "value"])
                for item in values:
                    labels = json.dumps(item.get("metric", {}), sort_keys=True)
                    for timestamp, value in item.get("values", []):
                        writer.writerow([metric, labels, timestamp, value])

        return directory
