"""USD per million tokens, used for the budget guard and the cost panel.

Unknown models fall back to the mid-tier rate rather than reporting zero - a run
that silently reports $0.00 is worse than one that reports an approximation.
"""

from __future__ import annotations

# (input, output, cache_read) per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (3.00, 15.00, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}

_FALLBACK = PRICES["claude-sonnet-5"]


def _rates(model: str) -> tuple[float, float, float]:
    for prefix, rates in PRICES.items():
        if model.startswith(prefix):
            return rates
    return _FALLBACK


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float:
    inp, out, cached = _rates(model)
    total = (
        input_tokens * inp + output_tokens * out + cache_read_tokens * cached
    ) / 1_000_000
    return round(total, 6)
