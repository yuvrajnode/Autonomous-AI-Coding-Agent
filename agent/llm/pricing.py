"""USD per million tokens, used for the budget guard and the cost panel.

Unknown models fall back to the mid-tier rate rather than reporting zero - a run
that silently reports $0.00 is worse than one that reports an approximation.
"""

from __future__ import annotations

# (input, output, cache_read) per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00, 1.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-opus-4-7": (5.00, 25.00, 0.50),
    "claude-opus-4-6": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-sonnet-4": (3.00, 15.00, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}

_FALLBACK = PRICES["claude-sonnet-5-5"]


def _rates(model: str) -> tuple[float, float, float]:
    # Longest prefix wins, so `claude-opus-5-5` is not billed as `claude-opus-5`.
    matches = [prefix for prefix in PRICES if model.startswith(prefix)]
    if not matches:
        return _FALLBACK
    return PRICES[max(matches, key=len)]


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float:
    inp, out, cached = _rates(model)
    total = (
        input_tokens * inp + output_tokens * out + cache_read_tokens * cached
    ) / 1_000_000
    return round(total, 6)
