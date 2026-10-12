"""Prometheus adapter for the HUST vLLM fork used by GenNova."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from probing.ext.engines.prometheus import (
    PrometheusSample,
    metric_suffix,
    parse_prometheus_text,
)
from probing.ext.engines.sglang import build_metrics_url

DEFAULT_METRICS_PATH = "/metrics"


@dataclass(frozen=True)
class VllmHustMetricsSnapshot:
    engine_id: str
    metrics_url: str
    captured_at_ns: int
    raw_samples: tuple[PrometheusSample, ...]
    normalized: dict[str, float]
    scrape_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "metrics_url": self.metrics_url,
            "normalized": self.normalized,
            "raw_count": len(self.raw_samples),
            "scrape_error": self.scrape_error,
        }


def _series(
    samples: Iterable[PrometheusSample], metric_name: str
) -> dict[tuple[tuple[str, str], ...], float]:
    wanted = metric_name.lower()
    return {
        tuple(sorted(sample.labels.items())): sample.value
        for sample in samples
        if metric_suffix(sample.name).lower() == wanted
    }


def _gauge_sum(samples: Iterable[PrometheusSample], metric_name: str) -> float | None:
    series = _series(samples, metric_name)
    return sum(series.values()) if series else None


def _gauge_max(samples: Iterable[PrometheusSample], metric_name: str) -> float | None:
    series = _series(samples, metric_name)
    return max(series.values()) if series else None


def _counter_delta(
    current_samples: Iterable[PrometheusSample],
    previous_samples: Iterable[PrometheusSample],
    metric_name: str,
) -> float | None:
    current = _series(current_samples, metric_name)
    previous = _series(previous_samples, metric_name)
    if not current or current.keys() != previous.keys():
        return None
    deltas = [current[key] - previous[key] for key in current]
    if any(delta < 0 for delta in deltas):
        return None
    return sum(deltas)


def _histogram_window_average(
    current_samples: Iterable[PrometheusSample],
    previous_samples: Iterable[PrometheusSample],
    base_name: str,
) -> float | None:
    total = _counter_delta(current_samples, previous_samples, f"{base_name}_sum")
    count = _counter_delta(current_samples, previous_samples, f"{base_name}_count")
    if total is None or count is None or count <= 0:
        return None
    return total / count


def normalize_vllm_hust_samples(
    samples: list[PrometheusSample],
    *,
    previous_samples: list[PrometheusSample]
    | tuple[PrometheusSample, ...]
    | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, float]:
    """Map vllm-hust samples to canonical metrics.

    Gauges are safe on the first scrape. Rates, latency histograms, and cache
    hit ratio are emitted only from a valid two-scrape window. A counter reset
    or label-set change invalidates that derived metric for the current window.
    """

    normalized: dict[str, float] = {}

    inflight = _gauge_sum(samples, "num_requests_running")
    if inflight is not None:
        normalized["inflight_requests"] = inflight

    queue = _gauge_sum(samples, "num_requests_waiting")
    if queue is not None:
        normalized["queue_depth"] = queue

    kv_usage = _gauge_max(samples, "kv_cache_usage_perc")
    if kv_usage is not None and 0 <= kv_usage <= 1:
        normalized["kv_cache_usage_ratio"] = kv_usage

    if previous_samples is None or elapsed_seconds is None or elapsed_seconds <= 0:
        return normalized

    tokens = _counter_delta(samples, previous_samples, "generation_tokens_total")
    if tokens is not None:
        normalized["throughput_tps"] = tokens / elapsed_seconds

    ttft_seconds = _histogram_window_average(
        samples, previous_samples, "time_to_first_token_seconds"
    )
    if ttft_seconds is not None:
        normalized["ttft_ms"] = ttft_seconds * 1000.0

    tpot_seconds = _histogram_window_average(
        samples,
        previous_samples,
        "request_time_per_output_token_seconds",
    )
    if tpot_seconds is not None:
        normalized["tpot_ms"] = tpot_seconds * 1000.0

    queries = _counter_delta(samples, previous_samples, "prefix_cache_queries_total")
    hits = _counter_delta(samples, previous_samples, "prefix_cache_hits_total")
    if queries is not None and queries > 0 and hits is not None:
        ratio = hits / queries
        if 0 <= ratio <= 1:
            normalized["cache_hit_ratio"] = ratio

    return normalized


def fetch_vllm_hust_metrics(
    engine_id: str,
    router_addr: str,
    *,
    metrics_path: str = DEFAULT_METRICS_PATH,
    timeout_seconds: float = 3.0,
    previous: VllmHustMetricsSnapshot | None = None,
) -> VllmHustMetricsSnapshot:
    metrics_url = build_metrics_url(router_addr, metrics_path)
    captured_at_ns = time.time_ns()
    request = urllib.request.Request(metrics_url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return VllmHustMetricsSnapshot(
            engine_id,
            metrics_url,
            captured_at_ns,
            tuple(),
            {},
            f"HTTP {exc.code}: {exc.reason}",
        )
    except Exception as exc:  # pragma: no cover - network errors vary
        return VllmHustMetricsSnapshot(
            engine_id, metrics_url, captured_at_ns, tuple(), {}, str(exc)
        )

    samples = parse_prometheus_text(body)
    elapsed_seconds = None
    previous_samples = None
    if previous is not None:
        elapsed_seconds = (captured_at_ns - previous.captured_at_ns) / 1_000_000_000
        previous_samples = previous.raw_samples
    return VllmHustMetricsSnapshot(
        engine_id=engine_id,
        metrics_url=metrics_url,
        captured_at_ns=captured_at_ns,
        raw_samples=tuple(samples),
        normalized=normalize_vllm_hust_samples(
            samples,
            previous_samples=previous_samples,
            elapsed_seconds=elapsed_seconds,
        ),
    )
