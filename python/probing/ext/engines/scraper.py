"""Persist scraped engine metrics and run background scrapers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from probing.core.table import table
from probing.ext.engines.registry import (
    EngineRegistration,
    get_engine,
    list_engines,
    update_scrape_result,
)
from probing.ext.engines.sglang import fetch_sglang_metrics, flatten_samples_for_storage
from probing.ext.engines.vllm_hust import (
    VllmHustMetricsSnapshot,
    fetch_vllm_hust_metrics,
)


@table("inference_engine_metric")
@dataclass
class InferenceEngineMetric:
    timestamp_ns: int
    engine_id: str
    engine_type: str
    metric_name: str
    metric_value: float
    labels: str


_scraper_lock = threading.Lock()
_window_lock = threading.Lock()
_scraper_started = False
_vllm_hust_windows: dict[str, tuple[int, VllmHustMetricsSnapshot]] = {}


def scrape_interval_seconds() -> float:
    raw = os.environ.get("PROBING_ENGINE_SCRAPE_INTERVAL", "5")
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 5.0


def scrape_engine(registration: EngineRegistration) -> dict:
    if registration.engine_type == "sglang":
        snapshot = fetch_sglang_metrics(
            registration.engine_id,
            registration.router_addr,
            metrics_path=registration.metrics_path,
        )
    elif registration.engine_type == "vllm-hust":
        with _window_lock:
            prior = _vllm_hust_windows.get(registration.engine_id)
        previous = (
            prior[1]
            if prior is not None and prior[0] == registration.registered_at_ns
            else None
        )
        snapshot = fetch_vllm_hust_metrics(
            registration.engine_id,
            registration.router_addr,
            metrics_path=registration.metrics_path,
            previous=previous,
        )
        if snapshot.scrape_error is None and snapshot.raw_samples:
            with _window_lock:
                _vllm_hust_windows[registration.engine_id] = (
                    registration.registered_at_ns,
                    snapshot,
                )
    else:
        update_scrape_result(
            registration.engine_id,
            normalized={},
            scrape_error=f"unsupported engine_type={registration.engine_type}",
        )
        return {"engine_id": registration.engine_id, "error": "unsupported engine_type"}
    update_scrape_result(
        registration.engine_id,
        normalized=snapshot.normalized,
        scrape_error=snapshot.scrape_error,
    )

    if snapshot.raw_samples:
        timestamp_ns = time.time_ns()
        rows = flatten_samples_for_storage(
            registration.engine_id,
            registration.engine_type,
            timestamp_ns,
            list(snapshot.raw_samples),
        )
        for row in rows:
            InferenceEngineMetric(*row).save()
        for metric_name, metric_value in snapshot.normalized.items():
            InferenceEngineMetric(
                timestamp_ns=timestamp_ns,
                engine_id=registration.engine_id,
                engine_type=registration.engine_type,
                metric_name=f"normalized.{metric_name}",
                metric_value=metric_value,
                labels="normalized=1",
            ).save()

    return snapshot.to_dict()


def scrape_all() -> list[dict]:
    results = []
    for registration in list_engines():
        results.append(scrape_engine(registration))
    return results


def _scraper_loop() -> None:
    while True:
        try:
            scrape_all()
        except Exception:
            pass
        time.sleep(scrape_interval_seconds())


def ensure_scraper_running() -> None:
    global _scraper_started
    if os.environ.get("PROBING_ENGINE_SCRAPE", "1").strip().lower() in {
        "0",
        "false",
        "off",
    }:
        return
    with _scraper_lock:
        if _scraper_started:
            return
        thread = threading.Thread(
            target=_scraper_loop, name="probing-engine-scraper", daemon=True
        )
        thread.start()
        _scraper_started = True
