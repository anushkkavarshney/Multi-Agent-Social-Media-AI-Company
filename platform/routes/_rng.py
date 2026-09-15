"""Process-wide engagement RNG shared by the publish and tick routes.

Day-1 bug (fixed 2026-09-15, before Day 3's "genuinely noisy week"): both
routes/posts.py::_rng() and routes/simulation.py::_rng() re-seeded
``np.random.default_rng(settings.sim_seed)`` on EVERY call. Every accrue
therefore drew from the IDENTICAL random stream — posts differed only through
the deterministic hidden-rule multipliers, and two consecutive /simulate/tick
calls produced the same noise. That silently erased the statistical noise the
engagement model depends on.

Fix: ONE Generator per process, seeded once (from settings.sim_seed) at first
use. Its stream advances across every publish and tick, so each post gets its
own noise draw while a whole process run stays reproducible from the same seed.

Tests call ``reset_rng()`` — exactly like they call reset_engine_for_tests() —
so a cleared DB + cleared RNG replays the same sequence. The /simulate/tick
route keeps its optional ``seed`` override: a caller that explicitly passes a
seed (deterministic single-tick tests) gets a fresh local generator and does
not consume the process stream.
"""

from __future__ import annotations

import numpy as np

from config.settings import get_settings

_rng: np.random.Generator | None = None


def get_rng() -> np.random.Generator:
    """The shared, advancing Generator; created lazily and seeded once."""
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(get_settings().sim_seed)
    return _rng


def reset_rng() -> None:
    """Drop the shared Generator so the next get_rng() re-seeds (tests only)."""
    global _rng
    _rng = None