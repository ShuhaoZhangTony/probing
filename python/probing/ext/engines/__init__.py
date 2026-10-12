"""Inference engine metrics for Probing-RL.

Generic registration is framework-neutral. Concrete adapters (SGLang, Slime
``sglang_router``, and vllm-hust) own runtime-specific normalization. Metrics
paths differ by adapter::

    /engine_metrics  — Slime sglang_router / model gateway
    /metrics         — sglang.launch_server or vllm-hust
"""

from probing.ext.engines.registry import (
    get_engine,
    list_engines,
    register_engine,
    register_slime_sglang_router,
    register_vllm_hust,
    unregister_engine,
)
from probing.ext.engines.scraper import (
    ensure_scraper_running,
    scrape_all,
    scrape_engine,
)

__all__ = [
    "ensure_scraper_running",
    "get_engine",
    "list_engines",
    "register_engine",
    "register_slime_sglang_router",
    "register_vllm_hust",
    "scrape_all",
    "scrape_engine",
    "unregister_engine",
]
