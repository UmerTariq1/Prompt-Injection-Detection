"""
stage0_cache.py
---------------
Stage 0 — Hash Cache

The fastest possible check: SHA-256 hash lookup in an LRU cache.
Previously-seen inputs return instantly (<0.01ms) with no ML inference.

Two buckets:
  - known_injections : prompts flagged as INJECTION by any downstream stage
  - known_safe       : prompts cleared as SAFE by any downstream stage

The pipeline calls `update()` after every verdict so the cache grows
organically over time without manual population.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

from utils.config import CFG
from utils.result import DetectionResult, Verdict

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache backed by an OrderedDict."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._store: OrderedDict[str, DetectionResult] = OrderedDict()

    def get(self, key: str) -> Optional[DetectionResult]:
        if key not in self._store:
            return None
        # Move to end (most-recently-used)
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: DetectionResult) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Cache evicted: %s", evicted_key[:16])

    def __len__(self) -> int:
        return len(self._store)


# Module-level cache instance (shared across all calls in the process)
_injection_cache = LRUCache(max_size=CFG.cache_max_size)
_safe_cache      = LRUCache(max_size=CFG.cache_max_size)


def _hash(text: str) -> str:
    """SHA-256 hex digest of the input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(text: str) -> Optional[DetectionResult]:
    """
    Look up the preprocessed text in the cache.

    Returns
    -------
    DetectionResult with stage=0 if found, else None.
    """
    t0  = time.perf_counter()
    key = _hash(text)

    hit = _injection_cache.get(key) or _safe_cache.get(key)
    if hit is None:
        return None

    latency = (time.perf_counter() - t0) * 1000
    return DetectionResult(
        verdict=hit.verdict,
        score=hit.score,
        stage=0,
        reason=f"Cache hit (originally detected at Stage {hit.stage})",
        latency_ms=latency,
    )


def update(text: str, result: DetectionResult) -> None:
    """
    Store a verdict in the appropriate cache bucket so future
    identical inputs skip all detection stages.
    """
    key = _hash(text)
    if result.verdict == Verdict.INJECTION:
        _injection_cache.put(key, result)
    elif result.verdict == Verdict.SAFE:
        _safe_cache.put(key, result)
    # UNCERTAIN results are not cached — they should be re-evaluated


def cache_stats() -> dict:
    return {
        "injection_cache_size": len(_injection_cache),
        "safe_cache_size":      len(_safe_cache),
        "max_size":             CFG.cache_max_size,
    }
