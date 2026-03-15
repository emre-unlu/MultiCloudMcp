import csv
import os
import tempfile
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin

from .observability_helpers import DEFAULT_JAEGER_URL, http_json

if TYPE_CHECKING:
    from fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.tool()
    def get_traces(namespace: str, duration: int) -> str:
        """Collect trace data from the service using Jaeger."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration)
        directory = tempfile.mkdtemp(prefix="traces_")

        services_url = urljoin(DEFAULT_JAEGER_URL, "/api/services")
        services_response = http_json(services_url)
        services = services_response.get("data", [])
        if not services:
            raise RuntimeError("Jaeger returned no services to query.")

        filename = os.path.join(directory, "traces.csv")
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["namespace", "service", "trace_id", "span_count", "start_time_us", "duration_us"])

            for service in services:
                params = {
                    "service": service,
                    "start": int(start_time.timestamp() * 1_000_000),
                    "end": int(end_time.timestamp() * 1_000_000),
                    "limit": 100,
                }
                url = urljoin(DEFAULT_JAEGER_URL, "/api/traces") + "?" + urlencode(params)
                traces_response = http_json(url)
                for trace in traces_response.get("data", []):
                    trace_id = trace.get("traceID")
                    spans = trace.get("spans", [])
                    span_count = len(spans)
                    if spans:
                        span_start = min(span.get("startTime", 0) for span in spans)
                        span_end = max(span.get("startTime", 0) + span.get("duration", 0) for span in spans)
                        duration_us = span_end - span_start
                    else:
                        span_start = 0
                        duration_us = 0
                    writer.writerow([namespace, service, trace_id, span_count, span_start, duration_us])

        return directory
