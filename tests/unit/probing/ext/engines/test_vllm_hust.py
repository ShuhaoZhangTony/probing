from __future__ import annotations

import json
from types import SimpleNamespace

from probing.ext.engines.prometheus import parse_prometheus_text
from probing.ext.engines.registry import (
    get_engine,
    register_vllm_hust,
    unregister_engine,
)
from probing.ext.engines.vllm_hust import normalize_vllm_hust_samples
from probing.ext.engines.vllm_hust import VllmHustMetricsSnapshot


def _samples(
    *,
    running: tuple[float, float] = (2, 1),
    waiting: tuple[float, float] = (3, 4),
    kv: tuple[float, float] = (0.4, 0.7),
    tokens: tuple[float, float] = (100, 200),
    ttft_sum: tuple[float, float] = (10, 20),
    ttft_count: tuple[float, float] = (10, 20),
    tpot_sum: tuple[float, float] = (5, 15),
    tpot_count: tuple[float, float] = (50, 150),
    queries: tuple[float, float] = (100, 200),
    hits: tuple[float, float] = (40, 120),
):
    rows = []
    for index, engine in enumerate(("0", "1")):
        labels = f'{{model_name="Qwen3.5-35B",engine="{engine}"}}'
        values = {
            "num_requests_running": running[index],
            "num_requests_waiting": waiting[index],
            "kv_cache_usage_perc": kv[index],
            "generation_tokens_total": tokens[index],
            "time_to_first_token_seconds_sum": ttft_sum[index],
            "time_to_first_token_seconds_count": ttft_count[index],
            "request_time_per_output_token_seconds_sum": tpot_sum[index],
            "request_time_per_output_token_seconds_count": tpot_count[index],
            "prefix_cache_queries_total": queries[index],
            "prefix_cache_hits_total": hits[index],
        }
        rows.extend(f"vllm:{name}{labels} {value}" for name, value in values.items())
    return parse_prometheus_text("\n".join(rows))


def test_first_scrape_only_emits_instantaneous_gauges():
    normalized = normalize_vllm_hust_samples(_samples())

    assert normalized == {
        "inflight_requests": 3,
        "queue_depth": 7,
        "kv_cache_usage_ratio": 0.7,
    }


def test_second_scrape_derives_rates_and_window_latencies():
    previous = _samples()
    current = _samples(
        running=(4, 2),
        waiting=(1, 0),
        kv=(0.5, 0.8),
        tokens=(160, 260),
        ttft_sum=(13, 24),
        ttft_count=(12, 22),
        tpot_sum=(7, 19),
        tpot_count=(70, 190),
        queries=(140, 260),
        hits=(70, 160),
    )

    normalized = normalize_vllm_hust_samples(
        current,
        previous_samples=previous,
        elapsed_seconds=10,
    )

    assert normalized["inflight_requests"] == 6
    assert normalized["queue_depth"] == 1
    assert normalized["kv_cache_usage_ratio"] == 0.8
    assert normalized["throughput_tps"] == 12
    assert normalized["ttft_ms"] == 1750
    assert normalized["tpot_ms"] == 100
    assert normalized["cache_hit_ratio"] == 0.7


def test_counter_reset_omits_only_unreliable_derived_metric():
    previous = _samples()
    current = _samples(tokens=(10, 260), hits=(70, 160), queries=(140, 260))

    normalized = normalize_vllm_hust_samples(
        current,
        previous_samples=previous,
        elapsed_seconds=10,
    )

    assert "throughput_tps" not in normalized
    assert normalized["cache_hit_ratio"] == 0.7
    assert normalized["inflight_requests"] == 3


def test_register_vllm_hust_preserves_required_provenance_labels():
    engine_id = "test-vllm-hust"
    unregister_engine(engine_id)
    try:
        registration = register_vllm_hust(
            "127.0.0.1:8000",
            engine_id=engine_id,
            model_id="Qwen/Qwen3.5-35B-A3B",
            runtime_revision="ced6857",
            accelerator_backend_revision="5422a07",
            labels={"model_id": "must-not-override", "site": "lab"},
            auto_scrape=False,
        )

        assert registration.engine_type == "vllm-hust"
        assert registration.metrics_path == "/metrics"
        assert registration.metrics_url == "http://127.0.0.1:8000/metrics"
        assert registration.labels == {
            "site": "lab",
            "integration": "gennova",
            "model_id": "Qwen/Qwen3.5-35B-A3B",
            "runtime_revision": "ced6857",
            "accelerator_backend": "vllm-ascend-hust",
            "accelerator_backend_revision": "5422a07",
        }
        assert get_engine(engine_id) is registration
    finally:
        unregister_engine(engine_id)


def test_scraper_dispatches_vllm_hust_and_reuses_previous_window(monkeypatch):
    from probing.ext.engines import scraper

    engine_id = "test-vllm-hust-scraper"
    unregister_engine(engine_id)
    registration = register_vllm_hust(
        "127.0.0.1:8000",
        engine_id=engine_id,
        model_id="Qwen/Qwen3.5-35B-A3B",
        runtime_revision="ced6857",
        accelerator_backend_revision="5422a07",
        auto_scrape=False,
    )
    calls = []

    def fake_fetch(*args, previous=None, **kwargs):
        calls.append(previous)
        return VllmHustMetricsSnapshot(
            engine_id=engine_id,
            metrics_url="http://127.0.0.1:8000/metrics",
            captured_at_ns=100 + len(calls),
            raw_samples=tuple(_samples()),
            normalized={"inflight_requests": 3.0},
        )

    monkeypatch.setattr(scraper, "fetch_vllm_hust_metrics", fake_fetch)
    monkeypatch.setattr(scraper.InferenceEngineMetric, "save", lambda self: None)
    try:
        scraper.scrape_engine(registration)
        scraper.scrape_engine(registration)
        assert calls[0] is None
        assert isinstance(calls[1], VllmHustMetricsSnapshot)
        assert get_engine(engine_id).last_normalized == {"inflight_requests": 3.0}
    finally:
        unregister_engine(engine_id)


def test_http_registration_requires_vllm_hust_provenance():
    from probing.handlers.pythonext import register_inference_engine

    payload = json.loads(
        register_inference_engine("127.0.0.1:8000", engine_type="vllm-hust")
    )
    assert payload == {
        "error": (
            "vllm-hust registration requires model_id, runtime_revision, "
            "and accelerator_backend_revision"
        )
    }


def test_http_registration_routes_to_vllm_hust_adapter(monkeypatch):
    import probing.ext.engines
    from probing.handlers.pythonext import register_inference_engine

    calls = []
    registration = SimpleNamespace(to_dict=lambda: {"engine_type": "vllm-hust"})

    def fake_register(router_addr, **kwargs):
        calls.append((router_addr, kwargs))
        return registration

    monkeypatch.setattr(probing.ext.engines, "register_vllm_hust", fake_register)
    monkeypatch.setattr(
        probing.ext.engines,
        "scrape_engine",
        lambda value: {"engine_id": "qwen35", "normalized": {}},
    )
    payload = json.loads(
        register_inference_engine(
            "127.0.0.1:8000",
            engine_id="qwen35",
            engine_type="vllm-hust",
            model_id="Qwen/Qwen3.5-35B-A3B",
            runtime_revision="ced6857",
            accelerator_backend_revision="5422a07",
        )
    )

    assert calls[0][0] == "127.0.0.1:8000"
    assert calls[0][1]["model_id"] == "Qwen/Qwen3.5-35B-A3B"
    assert calls[0][1]["accelerator_backend_revision"] == "5422a07"
    assert payload["engine"]["engine_type"] == "vllm-hust"
